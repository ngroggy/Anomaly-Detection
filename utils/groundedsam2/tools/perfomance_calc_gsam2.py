import json
import argparse


def evaluate(output_json_path, detections_json_path):
    # Load the output.json file (your ground truth anomalies data)
    with open(output_json_path, 'r') as file:
        output_data = json.load(file)

    # Load the detections.json file (your model predictions)
    with open(detections_json_path, 'r') as file:
        detections_data = json.load(file)

    # Initialize counters for the evaluation
    total_images = 0
    correct_predictions = 0
    incorrect_predictions_1_to_0 = 0
    incorrect_predictions_0_to_1 = 0

    # Iterate through the output data and compare with the detections
    for entry in output_data:
        image_id = entry['image_id']
        actual_anomaly = entry['anomaly']

        predicted_anomaly = None

        # Check if there is a corresponding prediction in detections.json
        for detection_dict in detections_data:
            if detection_dict["image_path"] == "data/" + image_id:
                predicted_anomaly = len(detection_dict['annotations']) > 0
                break
        
        if predicted_anomaly is not None:  # If there's a prediction
            total_images += 1
            if actual_anomaly == predicted_anomaly:
                correct_predictions += 1
            elif actual_anomaly == 1 and predicted_anomaly == 0:
                incorrect_predictions_1_to_0 += 1
            elif actual_anomaly == 0 and predicted_anomaly == 1:
                incorrect_predictions_0_to_1 += 1

    return total_images, correct_predictions, incorrect_predictions_1_to_0, incorrect_predictions_0_to_1


if __name__ == '__main__':
    # output_json_path = '/home/ngrogg/github/output/output_modified.json'
    # detections_json_path = "/home/ngrogg/github/Anomaly-Detection/outputs/GroundedSAM2/03/crop/predictions.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth-json", default=None, required=True)
    parser.add_argument("--detections-json", default=None, required=True)

    args = parser.parse_args()

    groundtruth_json_path = args.groundtruth_json
    detections_json_path = args.detections_json
    # Evaluate the model
    total_images, correct_predictions, incorrect_predictions_1_to_0, incorrect_predictions_0_to_1 = evaluate(
        groundtruth_json_path,
        detections_json_path
    )
    # Calculate the percentage of correct predictions
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
    else:
        accuracy = 0

    # Print the results
    print(f"Total images compared: {total_images}")
    print(f"Correct predictions: {correct_predictions} ({accuracy:.2f}%)")
    print(f"Incorrect predictions (1 -> 0): {incorrect_predictions_1_to_0}")
    print(f"Incorrect predictions (0 -> 1): {incorrect_predictions_0_to_1}")

