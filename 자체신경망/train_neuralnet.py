import numpy as np
from mnist_reader import load_mnist
from two_layer_net import TwoLayerNet
from multi_layer_net import MultiLayerNet
import matplotlib.pyplot as plt

import mnist_reader
x_train, t_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, t_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

x_train = x_train / 255.0
x_test = x_test / 255.0

# 2층 신경망
network = MultiLayerNet(input_size = 784, hidden_size_list=[93], output_size=10)

# 하이퍼파라미터 
iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.303550152648054

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)



for i in range(iters_num):
    # 미니 배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)


    for key in('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 100번에 한 번씩 진행 상황 출력 (선택 사항)
    # if i % 100 == 0:
    #     # print(f"Iteration {i} / {iters_num}, Loss: {loss}")
    #     print(f"Iteration {i} / {iters_num}")


    # 1에포크당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(i ,":" , round(train_acc,4), round(test_acc, 4))

    
# plt.plot(train_loss_list, loss)
# plt.show
# print("학습 완료! 그래프를 그립니다...")
# plt.subplot(1, 2, 1)
# plt.plot(train_loss_list) # Y축 값만 리스트로 넣으면 X축은 자동으로 인덱스(0, 1, 2...)가 됩니다.
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.grid(True)
# plt.show() # 괄호()를 붙여서 함수를 실행해야 합니다.

# # 오른쪽 그래프 (정확도 그래프 - Epoch 기준)
# plt.subplot(1, 2, 2)
# # X축은 자동으로 Epoch 0, 1, 2, ... (list의 인덱스)가 됩니다.
# plt.plot(train_acc_list, label="Train Accuracy")
# plt.plot(test_acc_list, label="Test Accuracy", linestyle="--") # 테스트 정확도는 점선으로 표시
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Train vs Test Accuracy")
# plt.ylim(0.0, 1.0)                   # Y축 범위를 0.0에서 1.0으로 고정
# plt.legend(loc='lower right')      # 범례 표시 (오른쪽 아래)
# plt.grid(True)


# # # 전체 그래프 창을 띄웁니다.
# # plt.tight_layout() # 그래프들이 겹치지 않게 자동 조정
# plt.show()