#hyperparameters
import numpy as np
import tensorflow as tf
from get_mips_data import get_mips_data
from preprocess_binary import append_csv_features
from mrs_model_binary import Mrs_Model
from num_passes_model import Passes_Model
from matplotlib import pyplot as plt
from skimage.transform import resize



def train(model, train_inputs, train_labels, predict_passes=False):
    nrows = train_labels.shape[0]
    nbatches = int(np.ceil(nrows/model.batch_size))
    accuracy = np.zeros(nbatches)
    total_under = np.zeros(nbatches)
    total_over = np.zeros(nbatches)
    if predict_passes:
        train_labels = np.array([row[1] for row in train_labels])
        # print("passes label")
        # print(train_labels)
    else:
        train_labels = np.array([row[0] for row in train_labels])
        # print("mrs label")
        # print(train_labels)

    for batch in range(0,nbatches):
        # For every batch, compute then descend the gradients for the model's weights
        start_idx = batch * model.batch_size
        inputs = train_inputs[start_idx:min((start_idx + model.batch_size), nrows)]
        labels = train_labels[start_idx:min((start_idx + model.batch_size), nrows)]
        with tf.GradientTape() as tape:
            probabilities = model.call(inputs)
            loss = model.loss(probabilities, labels)
            acc, under, over, residuals = model.accuracy(probabilities, labels)
            accuracy[batch] = acc
            total_under[batch] = under
            total_over[batch] = over
            # model.loss_list.append(loss.numpy())
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if predict_passes:
        print("Done! Train Passes accuracy: " + str(tf.reduce_mean(accuracy).numpy()))
    else:
        print("Done! Train # mRs accuracy: " + str(tf.reduce_mean(accuracy).numpy()))

def test(model, test_inputs, test_labels, predict_passes=False):
    nrows = test_labels.shape[0]
    nbatches = int(np.ceil(nrows / model.batch_size))
    accuracy = np.zeros(nbatches)
    total_under = np.zeros(nbatches)
    total_over = np.zeros(nbatches)

    if predict_passes:
        test_labels = np.array([row[1] for row in test_labels])
        total_residuals = np.zeros((nbatches, 10))
    else:
        test_labels = np.array([row[0] for row in test_labels])
        total_residuals = np.zeros((nbatches, 2))

    for batch in range(0, nbatches):
        start_idx = batch * model.batch_size
        inputs = test_inputs[start_idx:min((start_idx + model.batch_size), nrows)]
        labels = test_labels[start_idx:min((start_idx + model.batch_size), nrows)]
        probs = model.call(inputs)
        acc, under, over, residuals = model.accuracy(probs, labels)
        accuracy[batch] = acc
        total_under[batch] = under
        total_over[batch] = over
        total_residuals[batch] = residuals

    return tf.reduce_mean(accuracy).numpy(), tf.reduce_mean(total_under).numpy(), tf.reduce_mean(total_over).numpy(), tf.reduce_mean(total_residuals, axis=0).numpy()

def metrics(model, test_inputs, test_labels, predict_tici=False, title="Confusion Matrix", categories=None, map=None):
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    from sklearn import metrics
    nrows = test_labels.shape[0]
    nbatches = int(np.ceil(nrows / model.batch_size))
    y_pred = []
    y_label = []
    # The list used to save the output probabilities
    y_prob = []
    # The list that store the probabilities
    # y_prob = []

    if predict_tici:
        test_labels = np.array([row[1] for row in test_labels])
        total_residuals = np.zeros((nbatches, 7))
    else:
        test_labels = np.array([row[0] for row in test_labels])
        total_residuals = np.zeros((nbatches, 2))

    for batch in range(0, nbatches):
        start_idx = batch * model.batch_size
        inputs = test_inputs[start_idx:min((start_idx + model.batch_size), nrows)]
        labels = test_labels[start_idx:min((start_idx + model.batch_size), nrows)]
        probs = model.call(inputs)
        # Print the probability
        # print("The probabilities are:")
        # print(probs)
        # print(type(probs))
        labels = labels.tolist()
        pred_y = []
        pred_prob = []
        for i in range(len(labels)):
            predicted = np.argmax(probs[i])
            pred_y.append(predicted)
            pred_prob.append(probs[i][1])
        y_label.extend(labels)
        y_pred.extend(pred_y)
        y_prob.extend(pred_prob)
    print(classification_report(y_label, y_pred))
    cm = confusion_matrix(y_label, y_pred)
    print(title)
    print(cm)
    roc_score = metrics.roc_auc_score(y_label, y_prob)
    print("The ROC score measured with SKlearn metrics is: ", round(roc_score, 3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    spec, sens, thresholds = metrics.roc_curve(y_label, y_prob)
    print(y_label)
    print(y_prob)
    plot2 = plt.figure(2, figsize=(10 * 0.75, 10.5 * 0.75))
    # plt.plot(spec, sens, label="Our model: ROC Score={}".format(round(roc_score, 3)))
    plt.plot(spec, sens, label="mRS ≤ 2 vs mRS > 2: AUC ={}".format(round(roc_score, 3), ), color="red",linewidth=2)
    # plt.plot([1, 0], [1, 0], linestyle="--", label="No skill", color="black",linewidth=2)
    plt.plot([1, 0], [1, 0], linestyle="--",  color="black",linewidth=2)
    # plt.plot([1, 0], [1, 0], linestyle="--")
    # plt.title('ROC Curve')
    plt.title('Receiver Operating Characteristic Curve \n for 30-day mRS Binary Outcome Classification')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.xlim(-0.002, 1)  # 设置x轴的范围
    plt.ylim(-0.00, 1.05)
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig('mrs.png')

    return

def main():
    num_mrs_scores = 2
    max_passes = 10 # last class represents >=10 passes
    num_epochs = 20 # epochs that i've found good are 4, 5, 6, 8 -Chris

    train_data, train_labels, test_data, test_labels, vocab_size = append_csv_features(get_mips_data())
    np.random.seed(0)
    # for ratio in range(0.1,1,0,1):
    ratio = 1
    train_num = int(train_data.shape[0]*ratio)
    replace_sample_index = np.random.choice(a=np.arange(train_data.shape[0]), size=train_num, replace=False)
    train_data = train_data[replace_sample_index]
    train_labels = train_labels[replace_sample_index]

    mrs_model = Mrs_Model(vocab_size, num_mrs_scores)
    passes_model = Passes_Model(vocab_size, max_passes)

    for i in range(num_epochs):
        print(i)
        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data = train_data[indices]
        train_labels = train_labels[indices]
        train(passes_model, train_data, train_labels, predict_passes=True)
        train(mrs_model, train_data, train_labels)

    passes_acc, passes_over, passes_under, passes_residual = test(passes_model, test_data, test_labels, predict_passes=True)
    mrs_acc, mrs_over, mrs_under, mrs_residual = test(mrs_model, test_data, test_labels)
    print("Passes Score Test Accuracy: " + str(passes_acc))
    print("Passes Overpredict Rate: " + str(passes_over))
    print("Passes Underpredict Rate: " + str(passes_under))
    print("Passes Residual: " + str(passes_residual))
    print("mRs Test Accuracy: " + str(mrs_acc))
    print("mRs Passes Overpredict Rate: " + str(mrs_over))
    print("mRs Passes Underpredict Rate: " + str(mrs_under))
    print("mRs Passes Residual : " + str(mrs_residual))
    metrics(mrs_model, test_data, test_labels, title="mRS prediction")
    # Save the trained model
    weight_path = "Weights_folder/Weights"

    mrs_model.save_weights(weight_path)



if __name__ == '__main__':
    main()
