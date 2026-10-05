"""The prices every cost in the paper is counted at.

API prices are the ratio of a cached input token, an uncached input token and
an output token. Infrastructure prices are in dollars per million tokens:
recomputing a missed token, and keeping a token in memory for a day.
"""

CACHED_PRICE = 1.0  # a cached input token
UNCACHED_PRICE = 10.0  # an uncached input token: ten times a cached one
OUTPUT_PRICE = 50.0  # an output token: fifty times a cached input token
RECOMPUTE_PRICE = 1.0  # $ per million missed tokens: what re-prefilling them costs to serve
STORAGE_PRICE_PER_DAY = 30.0  # $ per million tokens kept for a day: DRAM for a 70 B model's KV cache
TOKENS_PER_MILLION = 1e6
SECONDS_PER_DAY = 86_400
