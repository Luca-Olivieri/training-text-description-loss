import sys
from pathlib import Path

root_path = Path("/home/olivieri/exp").resolve()
src_path = root_path / "src"
sys.path.append(f"{str(src_path)}")

from data import *

def test_get_one_answer_gt_withState():

    obj = get_one_answer_gt("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", 5, return_state=True)

    assert list(obj.keys())[0] == "state"
    assert 5 in obj.keys()

def test_get_one_answer_gt_noState():

    obj = get_one_answer_gt("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", 5, return_state=False)

    assert list(obj.keys())[0] != "state"
    assert 5 in obj.keys()

def test_get_one_answer_pr_withState():

    obj = get_one_answer_pr("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", "llm_judge_assessment", "ConcatMasks_Ovr_Vr", 5, return_state=True)

    assert list(obj.keys())[0] == "state"
    assert 5 in obj.keys()

def test_get_one_answer_pr_noState():

    obj = get_one_answer_pr("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", "llm_judge_assessment", "ConcatMasks_Ovr_Vr", 5, return_state=False)

    assert list(obj.keys())[0] != "state"
    assert 5 in obj.keys()

def test_get_many_answer_gt_withState():

    objs = get_many_answer_gt("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", return_state=True)

    assert list(objs.keys())[0] == "state"
    assert all(e in objs.keys() for e in range(0, 23))

def test_get_many_answer_gt_noState():

    objs = get_many_answer_gt("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", return_state=False)

    assert list(objs.keys())[0] != "state"
    assert all(e in objs.keys() for e in range(0, 23))

def test_get_many_answer_pr_withState():

    objs = get_many_answer_pr("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", "llm_judge_assessment", "ConcatMasks_Ovr_Vr", return_state=True)

    assert list(objs.keys())[0] == "state"
    all(k in range(0, 23) for k in list(objs.keys())[1:])

def test_get_many_answer_pr_noState():

    objs = get_many_answer_pr("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", "llm_judge_assessment", "ConcatMasks_Ovr_Vr", return_state=False)

    assert list(objs.keys())[0] != "state"
    all(k in range(0, 23) for k in list(objs.keys()))

def test_get_one_eval_gt_withState():

    obj = get_one_eval_gt("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", 5, return_state=True)

    assert list(obj.keys())[0] == "state"
    assert 5 in obj.keys()

def test_get_one_eval_gt_noState():

    obj = get_one_eval_gt("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", 5, return_state=False)

    assert list(obj.keys())[0] != "state"
    assert 5 in obj.keys()

def test_get_one_eval_pr_withState():

    obj = get_one_eval_pr("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", "llm_judge_assessment", "1_original", 5, return_state=True)

    assert list(obj.keys())[0] == "state"
    assert 5 in obj.keys()

def test_get_one_eval_pr_noState():

    obj = get_one_eval_pr("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", "llm_judge_assessment", "1_original", 5, return_state=False)

    assert list(obj.keys())[0] != "state"
    assert 5 in obj.keys()

def test_get_many_eval_gt_withState():

    objs = get_many_eval_gt("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", return_state=True)

    assert list(objs.keys())[0] == "state"
    assert all(k in range(0, 23) for k in list(objs.keys())[1:])

def test_get_many_eval_gt_noState():

    objs = get_many_eval_gt("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", return_state=False)

    assert list(objs.keys())[0] != "state"
    all(k in range(0, 23) for k in list(objs.keys())[1:])

def test_get_many_eval_pr_withState():

    objs = get_many_eval_pr("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", "llm_judge_assessment", "1_original", return_state=True)

    assert list(objs.keys())[0] == "state"
    all(k in range(0, 23) for k in list(objs.keys()))

def test_get_many_eval_pr_noState():

    objs = get_many_eval_pr("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", "llm_judge_assessment", "1_original", return_state=False)

    assert list(objs.keys())[0] != "state"
    all(k in range(0, 23) for k in list(objs.keys())[1:])

if __name__ == "__main__":
    objs = get_many_eval_pr("LRASPP_MobileNet_V3", "letterboxed", "points", "non-splitted", "llm_judge_assessment", "1_original", return_state=False)
    print(objs)
