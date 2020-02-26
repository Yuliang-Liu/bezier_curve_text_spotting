from .word_eval import do_coco_evaluation


def word_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    rec_type,
    expected_results,
    expected_results_sigma_tol,
):
    return do_coco_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        rec_type=rec_type,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
