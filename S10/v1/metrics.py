from sklearn.metrics import ConfusionMatrixDisplay

def get_metrics(labels_df, class_labels):
    ConfusionMatrixDisplay.from_predictions(
        labels_df.prediction.values, labels_df.target.values,
        display_labels=class_labels, text_kw={'fontsize':5},
        xticks_rotation=90,
        )

    correct_pred = {classname: 0 for classname in class_labels}
    total_pred = {classname: 0 for classname in class_labels}

    for label, prediction in zip(labels_df.target.values,
        labels_df.prediction.values):
        if label == prediction:
            correct_pred[class_labels[label]] += 1
        total_pred[class_labels[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')