"""Label-free source-exemplar linkage; source labels enter only at transform.

Within each Flight Distance, EM estimates per-field copy rates and a null
mixture. Exact agreement is weighted by its background probability. All source
rows in that distance group are considered; there is no nearest-neighbor cap.
The generative hypothesis is a diagnostic model, not an asserted data origin.
"""
import numpy as np
import pandas as pd

FD = 'Flight Distance'

class ExemplarLinkage:
    def __init__(self, sample_rows=100000, iterations=10, seed=20261026):
        self.sample_rows, self.iterations, self.seed = sample_rows, iterations, seed

    def _encode(self, raw):
        return np.column_stack([pd.Categorical(raw[c].fillna('__MISSING__').astype(str),
                                              categories=self.vocab[j]).codes
                                for j,c in enumerate(self.columns)]).astype('int32')

    def _background(self, codes, distance):
        n = len(codes)
        fdcount = self.fd_counts.reindex(distance).fillna(0).to_numpy()
        result = np.empty((n,len(self.columns)), 'float32')
        for j in range(len(self.columns)):
            valuecount = self.global_counts[j].reindex(codes[:,j]).fillna(0).to_numpy()
            global_p = (valuecount+.5)/(self.n_train+.5*(len(self.vocab[j])+1))
            keys = pd.MultiIndex.from_arrays([distance,codes[:,j]])
            joint = self.joint_counts[j].reindex(keys).fillna(0).to_numpy()
            result[:,j] = (joint+20*global_p)/(fdcount+20)
        assert np.isfinite(result).all() and (result > 0).all()
        return result

    def _pairs(self, codes, distance):
        counts = np.array([len(self.source_groups.get(fd, ())) for fd in distance], 'int64')
        assert (counts > 0).all(), 'Caller must handle unmatched distances separately'
        starts = np.r_[0,np.cumsum(counts)[:-1]]
        rows = np.repeat(np.arange(len(codes),dtype='int32'),counts)
        source = np.empty(len(rows), 'int32')
        groups = pd.Series(np.arange(len(distance))).groupby(distance,sort=False).indices
        for fd, qi in groups.items():
            candidates = self.source_groups[fd]
            positions = starts[qi,None]+np.arange(len(candidates))[None,:]
            source[positions.ravel()] = np.tile(candidates,len(qi))
        bits = np.zeros(len(rows),'uint32')
        for j in range(len(self.columns)):
            bits |= (codes[rows,j] == self.source_codes[source,j]).astype('uint32') << j
        return rows, source, counts, starts, bits

    def _posterior(self, pairs, background):
        rows, source, counts, starts, bits = pairs
        p = self.copy_rates
        log_lr = np.full(len(rows),np.log1p(-p).sum(), 'float64')
        for j in range(len(p)):
            match = (bits & (1 << j)) != 0
            bonus = np.log1p(p[j]/((1-p[j])*background[:,j]))
            log_lr[match] += bonus[rows[match]]
        maximum = np.maximum(0,np.maximum.reduceat(log_lr,starts))
        weights = np.exp(log_lr-maximum[rows])*(self.link_prior/counts[rows])
        null = (1-self.link_prior)*np.exp(-maximum)
        denominator = np.add.reduceat(weights,starts)+null
        weights /= denominator[rows]
        null /= denominator
        evidence = maximum+np.log(denominator)
        return weights, null, evidence

    def fit(self, raw, source_raw):
        assert 'satisfaction' not in raw and 'id' not in raw
        assert 'satisfaction' not in source_raw and 'id' not in source_raw
        self.columns = [c for c in raw if c != FD]
        assert len(self.columns) == 20 and list(source_raw) == list(raw)
        self.n_train = len(raw)
        self.vocab = [pd.Index(pd.concat([raw[c],source_raw[c]],ignore_index=True)
                              .fillna('__MISSING__').astype(str).unique()) for c in self.columns]
        codes = self._encode(raw)
        self.source_codes = self._encode(source_raw)
        source_fd = source_raw[FD].to_numpy()
        self.source_groups = {fd:idx.astype('int32') for fd,idx in
                              pd.Series(np.arange(len(source_fd))).groupby(source_fd,sort=False).indices.items()}
        distance = raw[FD].to_numpy()
        self.fd_counts = pd.Series(distance).value_counts()
        self.global_counts, self.joint_counts = [], []
        for j in range(len(self.columns)):
            self.global_counts.append(pd.Series(codes[:,j]).value_counts())
            self.joint_counts.append(pd.DataFrame({'fd':distance,'value':codes[:,j]})
                                     .groupby(['fd','value'],sort=False).size())
        eligible = np.flatnonzero(np.isin(distance,list(self.source_groups)))
        rng = np.random.default_rng(self.seed)
        sample = np.sort(rng.choice(eligible,min(len(eligible),self.sample_rows),replace=False))
        bg = self._background(codes[sample],distance[sample])
        pairs = self._pairs(codes[sample],distance[sample])
        rows, _, _, _, bits = pairs
        self.copy_rates = np.full(len(self.columns),.3)
        self.link_prior = .5
        self.trace = []
        def objective(log_evidence):
            p = self.copy_rates
            return float(log_evidence.sum()+np.log(p).sum()+np.log1p(-p).sum()
                         +np.log(self.link_prior)+np.log1p(-self.link_prior))
        for iteration in range(self.iterations):
            weights, null, log_evidence = self._posterior(pairs,bg)
            linked = weights.sum()
            updated = np.empty_like(self.copy_rates)
            for j,p in enumerate(self.copy_rates):
                match = (bits & (1 << j)) != 0
                conditional_copy = p/(p+(1-p)*bg[:,j])
                expected_copy = np.sum(weights[match]*conditional_copy[rows[match]])
                updated[j] = np.clip((expected_copy+1)/(linked+2),.02,.98)
            self.trace.append(dict(iteration=iteration,penalized_log_likelihood=objective(log_evidence),
                                   link_prior=float(self.link_prior),copy_rates=self.copy_rates.tolist()))
            self.copy_rates = updated
            self.link_prior = float(np.clip((linked+1)/(len(sample)+2),.02,.98))
        _, _, log_evidence = self._posterior(pairs,bg)
        self.trace.append(dict(iteration=self.iterations,penalized_log_likelihood=objective(log_evidence),
                              link_prior=self.link_prior,copy_rates=self.copy_rates.tolist()))
        objective_path = [r['penalized_log_likelihood'] for r in self.trace]
        assert np.min(np.diff(objective_path)) > -1e-4
        self.fit_summary = dict(train_rows=len(raw),em_rows=len(sample),source_rows=len(source_raw),
                                em_pairs=len(rows),official_target_used=False,source_target_used_for_EM=False)
        return self

    def transform(self, raw, source_labels, batch_size=20000):
        labels = np.asarray(source_labels,dtype='float64')
        assert len(labels) == len(self.source_codes) and np.isin(labels,[0,1]).all()
        source_means = {fd:labels[idx].mean() for fd,idx in self.source_groups.items()}
        global_prior = labels.mean()
        answer = np.zeros((len(raw),7),'float32')
        for start in range(0,len(raw),batch_size):
            frame = raw.iloc[start:start+batch_size]
            distance = frame[FD].to_numpy()
            prior = np.array([source_means.get(fd,global_prior) for fd in distance])
            block = np.column_stack([prior,prior,np.zeros((len(frame),5))])
            eligible = np.flatnonzero(np.isin(distance,list(self.source_groups)))
            if len(eligible):
                codes = self._encode(frame.iloc[eligible])
                fd = distance[eligible]
                pairs = self._pairs(codes,fd)
                rows, source, counts, starts, _ = pairs
                weights, null, log_evidence = self._posterior(pairs,self._background(codes,fd))
                linked = np.add.reduceat(weights,starts)
                positive = np.add.reduceat(weights*labels[source],starts)
                conditional = np.divide(positive,linked,out=prior[eligible].copy(),where=linked>1e-15)
                maximum = np.maximum.reduceat(weights,starts)
                entropy = (-np.add.reduceat(weights*np.log(np.maximum(weights,1e-300)),starts)
                           -null*np.log(np.maximum(null,1e-300)))/np.log(counts+1)
                block[eligible] = np.column_stack([positive+null*prior[eligible],conditional,
                    linked,maximum,entropy,log_evidence,np.log1p(counts)])
            answer[start:start+len(frame)] = block
        assert np.isfinite(answer).all()
        return pd.DataFrame(answer,columns=['link_teacher','link_conditional_teacher','link_probability',
            'link_max_posterior','link_normalized_entropy','link_log_evidence','link_log_candidates'])
