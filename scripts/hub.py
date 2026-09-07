"""What `scripts/publish.py` and `scripts/fetch.py` have to agree on about the model hub.

Written down here rather than in either of them, since a repo one proposes to and the other pulls
from cannot be allowed to drift apart. Where the fetched files land is `paths.PRETRAINED`: that is
this project's own layout rather than the hub's.
"""

# The Hugging Face model repo the published checkpoints live in.
HF_REPO_ID = '446f6e6e79/CVAE-Suffix-Generation'
