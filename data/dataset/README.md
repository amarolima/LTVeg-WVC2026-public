# data/dataset/

Empty on purpose: the images are Google imagery and are not distributed.

Regenerate them ([docs/02_REGENERATING_THE_DATASET.md](../../docs/02_REGENERATING_THE_DATASET.md)) and
let the verifier fill this folder:

    python src/dataset/verify_dataset.py --images <regenerated_folder> --organize data/dataset

Resulting layout: `{train,val,test}/{OK,NOK}/*.png`. PNG files placed here are ignored by git.
