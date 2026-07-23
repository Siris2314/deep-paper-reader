import os


# Unit tests must not depend on the availability of the external vocabulary archive.
os.environ.setdefault("PWC_VOCAB_ALLOW_NETWORK", "false")
