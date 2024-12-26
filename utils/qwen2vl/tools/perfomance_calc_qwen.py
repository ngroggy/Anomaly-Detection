import json
import argparse

def evaluate(output_json_path, detections_json_path):
    # Load the output_modified.json (ground truth anomalies)
    with open(output_json_path, 'r') as file:
        output_modified_data = json.load(file)

    # Load the output_prompt.json (model predictions from the prompt)
    with open(detections_json_path, 'r') as file:
        output_prompt_data = json.load(file)

    # Initialize counters for the evaluation
    total_images = 0
    correct_predictions = 0
    incorrect_predictions_1_to_0 = 0
    incorrect_predictions_0_to_1 = 0

    # Iterate through the output_modified_data and compare with output_prompt_data
    for entry in output_modified_data:
        image_id = entry['image_id']
        actual_anomaly = entry['anomaly']
        
        # Check if there is a corresponding prediction in output_prompt.json
        predicted_text = output_prompt_data.get(image_id, None)
        
        # Convert the prediction text to 0 or 1
        predicted_anomaly = 0 if (predicted_text == "[]" or predicted_text == "[ ]") else 1 
        
        total_images += 1
        if actual_anomaly == predicted_anomaly:
            correct_predictions += 1
        elif actual_anomaly == 1 and predicted_anomaly == 0:
            incorrect_predictions_1_to_0 += 1
        elif actual_anomaly == 0 and predicted_anomaly == 1:
            incorrect_predictions_0_to_1 += 1

    return total_images, correct_predictions, incorrect_predictions_1_to_0, incorrect_predictions_0_to_1


if __name__ == '__main__':
    # output_modified_path = './output_modified.json'
    # output_prompt_path = "/home/ngrogg/Downloads/predictions(5).json"
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
