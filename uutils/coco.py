from pprint import pprint
from pycocotools.coco import COCO


<<<<<<< HEAD
def id2l(coco):
=======
def id2l(root):

    dset = "val"
    coco_path = f"{root}/COCOdataset2017"
    coco = COCO(f"{coco_path}/annotations/instances_val2017.json")
>>>>>>> 17670707f224613b6a2f604dd1cf533f9e32a559

    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)

    ids = [c["id"] for c in cats]
    labels = [c["name"] for c in cats]

    return {
        "id2l": {id: l for id, l in zip(ids, labels)},
        "l2id": {l:id for id, l in zip(ids, labels)},
    }

