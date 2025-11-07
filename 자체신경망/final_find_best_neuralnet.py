import numpy as np
import matplotlib.pyplot as plt
import random # ⭐️ random 임포트
import time   # ⭐️ time 임포트
from mnist_reader import load_mnist
from multi_layer_net import MultiLayerNet 
from collections import OrderedDict

# 1. 데이터 로드
print("데이터 로딩 중...")
x_train_all, t_train_all = load_mnist('data/fashion', kind='train')
x_test, t_test = load_mnist('data/fashion', kind='t10k')

# 2. 정규화
x_train_all = x_train_all / 255.0
x_test = x_test / 255.0 

# 3. 훈련 / 검증 데이터 분리
validation_rate = 0.20
validation_num = int(x_train_all.shape[0] * validation_rate) 
shuffle_mask = np.random.permutation(x_train_all.shape[0])
x_train_all = x_train_all[shuffle_mask]
t_train_all = t_train_all[shuffle_mask]
x_val = x_train_all[:validation_num]
t_val = t_train_all[:validation_num]
x_train = x_train_all[validation_num:] 
t_train = t_train_all[validation_num:] 

print(f"데이터 분할 완료: Train({x_train.shape[0]}), Validation({x_val.shape[0]})")

# -----------------------------------------------------------------
# 📌 4. 기본 학습 설정 (탐색용)
# -----------------------------------------------------------------
# ⭐️ 탐색 시에는 Epoch를 줄여서 (예: 5 Epoch) 빠르게 테스트합니다.
# 1 Epoch = 480 iter (48000 / 100)
iters_num_search = 10000 # ( 1회 시도 : 5 Epochs = 2400) 
batch_size = 100
train_size = x_train.shape[0]
iter_per_epoch = max(train_size / batch_size, 1)

def train_model(lr, hidden_size):
    """지정된 lr, hs로 모델을 학습시키고 최고 검증 정확도를 반환"""
    hidden_list = [hidden_size] # 2층 네트워크로 고정
    network = MultiLayerNet(input_size=784, 
                            hidden_size_list=hidden_list, 
                            output_size=10)
    
    max_val_acc_trial = 0.0 # 해당 테스트의 최고 정확도

    for i in range(1, iters_num_search + 1):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        for param_key in network.params.keys():
            if param_key.startswith('W') or param_key.startswith('b'):
                network.params[param_key] -= lr * grad[param_key]

        if i % iter_per_epoch == 0:
            val_acc = network.accuracy(x_val, t_val)
            max_val_acc_trial = max(max_val_acc_trial, val_acc)
            
    return max_val_acc_trial

# -----------------------------------------------------------------
# 5. 1단계: 탐색 (Wide Random Search)
# -----------------------------------------------------------------
print("\n========== 1단계: 넓은 탐색 (Random Search) 시작 ==========")
num_trials_s1 = 30 # 📌 1단계 탐색 횟수 (30~100 추천)

# ⭐️ 넓은 탐색 범위
lr_range_wide = (-4, -0.3) # (log) 10^-4 (0.0001) ~ 10^-0.3 (0.5)
hs_range_wide = (20, 150)  # (int) 20 ~ 150
results_s1 = [] # (val_acc, lr, hs) 저장

start_time_s1 = time.time()

for i in range(num_trials_s1):
    # ⭐️ 범위 안에서 매번 새로운 파라미터 무작위 추출
    lr = 10**np.random.uniform(lr_range_wide[0], lr_range_wide[1])
    hs = np.random.randint(hs_range_wide[0], hs_range_wide[1])
    
    key = f"[S1 {i+1}/{num_trials_s1}] lr={lr:.6f} hs={hs}"
    print(f"--- STARTING TEST: {key} ---")
    
    # 모델 학습 및 평가
    val_acc = train_model(lr, hs)
    
    print(f"--- TEST FINISHED: {key} | Max Acc: {val_acc:.4f} ---")
    results_s1.append((val_acc, lr, hs))

end_time_s1 = time.time()
print(f"1단계 탐색 완료. (소요 시간: {end_time_s1 - start_time_s1:.2f}초)")


# -----------------------------------------------------------------
# 6. 2단계: 최적 범위 도출
# -----------------------------------------------------------------
print("\n========== 2단계: 최적 범위 도출 ==========")

# 정확도(val_acc) 기준으로 내림차순 정렬
results_s1.sort(key=lambda x: x[0], reverse=True)

best_s1_result = results_s1[0]
print(f"🥇 1단계 최고 정확도: {best_s1_result[0]:.4f} (lr={best_s1_result[1]:.6f}, hs={best_s1_result[2]})")

# 1단계 상위 20%의 결과를 "대박" 조합으로 간주
top_n_percent = 0.20
top_n_count = int(num_trials_s1 * top_n_percent)
if top_n_count < 3: top_n_count = max(3, len(results_s1)) # 최소 3개
top_results = results_s1[:top_n_count]

# "대박" 조합들의 min/max 범위를 찾음
min_lr = min(res[1] for res in top_results)
max_lr = max(res[1] for res in top_results)
min_hs = min(res[2] for res in top_results)
max_hs = max(res[2] for res in top_results)

print(f"--- 2단계 정밀 탐색 범위 (상위 {top_n_percent*100}%) ---")
print(f"LR 범위: {min_lr:.6f} ~ {max_lr:.6f}")
print(f"HS 범위: {min_hs} ~ {max_hs}")


# -----------------------------------------------------------------
# 7. 3단계: 정밀 탐색 (Narrow Grid Search)
# -----------------------------------------------------------------
print("\n========== 3단계: 정밀 탐색 (Grid Search) 시작 ==========")
steps_s2 = 5 # 📌 좁은 범위에서 5x5 = 25회 촘촘하게 탐색
lr_range_narrow = np.linspace(min_lr, max_lr, steps_s2)
hs_range_narrow = np.linspace(min_hs, max_hs, steps_s2, dtype=int)

best_val_acc_s2 = 0.0
best_params_s2 = {}

start_time_s2 = time.time()
trial_count = 0

for lr in lr_range_narrow:
    for hs in hs_range_narrow:
        trial_count += 1
        key = f"[S2 {trial_count}/{steps_s2**2}] lr={lr:.6f} hs={hs}"
        print(f"--- STARTING TEST: {key} ---")

        val_acc = train_model(lr, hs)
        print(f"--- TEST FINISHED: {key} | Max Acc: {val_acc:.4f} ---")
        
        if val_acc > best_val_acc_s2:
            best_val_acc_s2 = val_acc
            best_params_s2 = {'learning_rate': lr, 'hidden_size': hs}

end_time_s2 = time.time()
print(f"3단계 정밀 탐색 완료. (소요 시간: {end_time_s2 - start_time_s2:.2f}초)")


# -----------------------------------------------------------------
# 8. 최종 결과 발표
# -----------------------------------------------------------------
print("\n========== ALL TESTS FINISHED ==========")
print(f"🥇 1단계 최고 정확도 (랜덤): {best_s1_result[0]:.4f}")
print(f"🥇 3단계 최고 정확도 (정밀): {best_val_acc_s2:.4f}")
print(f"🥇 최종 최적 파라미터: {best_params_s2}")


# -----------------------------------------------------------------
# 9. (보너스) 1단계 '지진파' 그래프 시각화
# -----------------------------------------------------------------
print("1단계 탐색(지진파) 그래프를 그립니다...")
lrs = [res[1] for res in results_s1]
accs = [res[0] for res in results_s1]
hss = [res[2] for res in results_s1] # 점의 크기(size)로 사용

plt.figure(figsize=(10, 6))
# x=학습률, y=정확도, s=은닉층크기, c=정확도, alpha=투명도
plt.scatter(lrs, accs, s=hss, c=accs, cmap='viridis', alpha=0.7)
plt.xscale('log') # ⭐️ X축을 로그 스케일로
plt.xlabel('Learning Rate (Log Scale)')
plt.ylabel('Validation Accuracy')
plt.title(f"Stage 1: Random Search Results ({num_trials_s1} trials)")
plt.colorbar(label='Accuracy')
plt.grid(True, which="both", ls="--")

# 2단계에서 찾은 "대박" 범위(녹색)와 최종 최고점(빨간색) 표시
plt.axvspan(min_lr, max_lr, color='green', alpha=0.1, label=f'Best Range (Top {top_n_percent*100}%)')
plt.scatter(best_s1_result[1], best_s1_result[0], 
            s=best_s1_result[2], color='red', 
            edgecolors='black', zorder=5, label='S1 Best')
plt.legend()
plt.show()