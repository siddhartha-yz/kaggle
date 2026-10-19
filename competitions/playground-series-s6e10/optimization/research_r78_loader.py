"""Deferred fixed transform of existing hash/ID-checked SAT-free pretext outputs."""
from research_r26_loader import auxiliary as probability_auxiliary
from research_r78_features import encode_auxiliary

def auxiliary(ids,test=False):
 return encode_auxiliary(probability_auxiliary(ids,test=test))
