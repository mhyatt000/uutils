from pprint import pprint
from pycocotools.coco import COCO


def id2l(coco):

    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)

    ids = [c["id"] for c in cats]
    labels = [c["name"] for c in cats]

    return {
        "id2l": {id: l for id, l in zip(ids, labels)},
        "l2id": {l:id for id, l in zip(ids, labels)},
    }

