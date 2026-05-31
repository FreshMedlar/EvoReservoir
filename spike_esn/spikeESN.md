# Brain-Inspired Spike Echo State Network Dynamics for Aero-engine Intelligent Fault Prediction

Mo-Ran Liu, Tao Sun, and Xi-Ming Sun, Senior Member, IEEE

Abstract-Aero-engine fault prediction aims to accurately predict the development trend of the future state of aero-engines, so as to diagnose faults in advance. Traditional aero-engine parameter prediction methods mainly use the nonlinear mapping relationship of time series data but generally ignore the adequate spatio-temporal features contained in aero-engine data. To this end, we propose a brain-inspired spike echo state network (Spike-ESN) model for aero-engine intelligent fault prediction, which is used to effectively capture the evolution process of aero-engine time series data in the framework of spatio-temporal dynamics. In the proposed approach, we design a data spike input layer based on Poisson distribution inspired by the spike neural encoding mechanism of biological neurons, which can extract the useful temporal characteristics in aero-engine sequence data. Then, the temporal characteristics are input into a spike reservoir through the current calculation method of spike accumulation in neurons, which projects the data into a high-dimensional sparse space. In addition, we use the ridge regression method to read out the internal state of the spike reservoir. Finally, the experimental results of aero-engine states prediction demonstrate the superiority and potential of the proposed method.

Index Terms—Aero-engine measurement, fault diagnosis, artificial neural networks, brain-inspired learning systems.

# I. INTRODUCTION

As the core power source of civil or military aircrafts, aero-engine is a highly complex and precise pneumatic thermal mechanical system, and its working state has a direct or indirect impact on the safety, reliability, economy and other performance problems of aircrafts [1]–[5]. Since the aero-engine works in harsh environments of high temperature, high pressure and strong vibration for a long time, the probability of failure increases accordingly, which affects the working performance of the aero-engine and even causes bad flight accidents. In fact, the surge fault is one of the most representative and destructive faults of aero-engines or aircrafts. When the aircraft flies under extreme conditions, the aero-engine system is affected by the external environment [6], which makes the output pressure of the compressor less than the downstream of the system and the backward flow of high-pressure gas. Accordingly, the aero-engine produces

This work was supported in part by the National Key R&D Program of China under Grant 2018YFB1700102, in part by the National Natural Science Foundation of China under Grants 61890920 and 61890921, and in part by the the China Postdoctoral Science Foundation funded project under Grant 2022TQ0179. (Corresponding author: T. Sun.)   
M.-R. Liu and X.-M. Sun are with the Key Laboratory of Intelligent Control and Optimization for Industrial Equipment of Ministry of Education, Dalian University of Technology, Dalian 116024, China (e-mail: liumoran@mail.dlut.edu.cn; sunxm@dlut.edu.cn).   
T. Sun is with the Department of Automation, Tsinghua University, Beijing 100084, China(e-mail: tao.sun.meng@gmail.com).

Figure 1. Schematic diagram of data driven fault diagnosis of an aero-engine system.

violent vibration and overtemperature at the hot end, which causes the aircraft to shutdown in the air. The research shows that aero-engine fault occurrence generally includes instability inception, rotating stall and surge [7], [8], where rotating stall is the precursor of surge (i.e., as shown in Figure 1). Since the aero-engine data has strong nonlinear and high-dimensional characteristics in actual operation, the fault diagnosis of an aero-engine generally relies on the empirical criteria of the crew and ground researchers. However, due to the subjectivity of users, it is often impossible to accurately predict faults using empirical estimation methods, so that the aircraft operation

fault cannot be judged and dealt with in time. Therefore, it is of great practical significance for the safe, stable and efficient operation of an aircraft to scientifically predict the potential fault factors of aero-engines and terminate the surge fault in the early stage.

The aero-engine fault diagnosis method mainly includes the method based on the mechanism model and the method based on the data analysis. In the early stage, the forecasting method based on the mechanism model was the main research idea of aero-engine fault diagnosis [9]. For example, paper [10] proposed a fault influence coefficient matrix method for aero-engine physical model, which is used in many traditional fault diagnosis systems. By using a fault map model, each state of the aero-engine in [11] is mapped to a point or a region, where researchers can use fault maps to qualitatively delineate areas and identify fault types according to empirical criteria. In addition, the diagnostic methods based on nonlinear Kalman filter algorithm are designed for the Gaussian noise environment and nonlinear properties of aero-engine parameters. For instance, a fuzzy adaptive unscented Kalman filter algorithm is developed in [12]. In the presence of gas path measurement uncertainty, paper [13] improved a Kalman filter by using a multi-step recursive estimation strategy and self-tuning buffers. According to the above fault diagnosis report based on the mechanism model, the real aero-engine is an extremely complex nonlinear system, so the fault diagnosis method based on mechanism model has high requirements on the design accuracy of the model.

The fault diagnosis methods based on data analysis may not require precise physical models but sufficient judgmental experience and historical data. So far, data-driven based intelligent fault diagnosis methods mainly include artificial neural network [14]–[17], support vector machine [18], extreme learning machine [19], etc. In particular, an extended least squares support vector machine in [20] is presented to measure the failure of aero-engines. Moreover, the extreme learning machine based on quantum behavioral particle swarm optimization is applied to the diagnosis of gas turbofan engines in [21]. In fact, the artificial neural networks have more advantages than above machine learning methods in the area of nonlinear prediction [22], [23]. For example, paper [24] utilized a fault diagnosis method based on the back propagation neural network for the failure of the aero-engine sensors, while the back propagation neural network method cannot memorize the past sequence data. Instead, paper [25] designed a long short-term memory neural network for fault diagnosis in the cases of complex environment [26]. However, the long short-term memory neural network can hardly extract the temporal features of the data, and the solution of its weight parameters requires an additional optimization algorithm.

Therefore, the above observations motivate the following problems naturally: 1) How to provide a data-driven intelligent diagnosis method to solve the fault prediction problem of aero-engines? 2) How to introduce a spike neural mechanism with brain-inspired learning to deeply extract the spatiotemporal feature of an aero-engine time series with multidimensional, multi-scaled and multi-modal characteristics? 3) How to preserve bio-interpretability, sparsity and memory of a

neural network model for aero-engines? Inspired by the above problems, since aero-engines with highly coupled subsystems are sensitive to complex working environments, the research on aero-engine fault prediction in the area of aero-engines is also an inevitable and interesting topic. As far as we know, adequately capturing spatio-temporal feature is always one of the most challenging topics in the field of fault diagnosis and data prediction, and there is not report on the aero-engine fault prediction of the so-called "brain-inspired spike echo state network" from the viewpoint of spatio-temporal features, sparsity and memory. This paper is motivated by the above discussions.

Overall, the main contributions of this work are summarized as follows:

- Inspired by the spike encoding mechanism in biological neurons, we propose a spike input layer that utilizes Poisson distribution to randomly spike encode the aeroengine time series data, which emphasizes a temporal dynamic characteristic.   
- In order to further extract the spatial dynamic features of the aero-engine data, we designed a spike reservoir that uses a current calculation method of spike accumulation in neurons, which can project the temporal characteristics of the spike sequence into a high-dimensional space.   
- A brain-inspired spike echo state network (Spike-ESN) model is designed for aero-engine intelligent fault prediction. Compared with the echo state network (ESN) model [27] and the traditional auto-regressive and moving average (ARMA) model [28], the proposed Spike-ESN model with both sparsity and memory can effectively capture the evolution process of aero-engine time series data in spatio-temporal space, which plays a key role in real-time prediction of aero-engine states. In addition, we have conducted a lot of experiments on the actual aero-engine datasets with faults to prove the effectiveness of Spike-ESN for aero-engine fault prediction.

The remainder of this article is organized as follows. Section II introduces the time series prediction problem of aero-engine fault diagnosis. Section III designs a brain-inspired spike echo state network with the spike neural mechanism. Some numerical simulation results are checked in Section IV. Main conclusions of the paper are given in Section V.

# II. PROBLEM PRELIMINARIES

According to the statistics of China in the early decade of the 21st century, $60\%$ of aircraft mechanical failure accidents are caused by the aero-engine fault, and more than $30\%$ of the daily maintenance costs of aircraft are generated by aero-engines. In order to ensure the safety and stability of aero-engines, it is indispensable to monitor the fault and predict the life of aero-engines. Generally, scientists predict failures through various methods of detecting and calculating trends in the operating state of an aero-engine system. In the actual situation, the fault diagnosis system judges the abnormal signals sent out before the aero-engine failure by studying

Figure 2. Schematic diagram of operating data of an aero-engine.

the characteristics of the system, so that the crew can take crucial measures for the corresponding signals to prevent the occurrence of faults.

For example, the aircraft data with faults from an Aero-engine Research Institute is shown in Figure 2, where $D_{8}$ is the ambient pressure, $T_{6}$ and $T_{1}$ are the exhaust temperature and the inlet temperature respectively, $\alpha_{1}$ and $\alpha_{2}$ are the compressor opening angles, $PLA$ is the power lever angle, $H$ is the flight height, $M$ represents the Mach number, $V$ represents the flight speed, $N_{2}$ and $N_{1}$ are high and low pressure rotor speeds respectively, signal indicates the occurrence of a fault. As can be seen from Figure 2, around the 200th data point, the ambient pressure $D_{8}$ dropped sharply. At the same time, the high-pressure rotor speed $N_{2}$ and the low-pressure rotor speed $N_{1}$ decreased significantly, while signal indicated that the aero-engine has a fault. Based on this, it is feasible to predict aircraft engine failures through data trends.

In fact, the fault prediction of aero-engines can be regarded as a time series prediction problem. Empirically, it is assumed that the statistical evaluation index data is represented as a set of univariate time series $\{z_i\in \mathbb{R}\}_{i = 1}^N$ , where $z_{i}$ is the $i^{th}$ point element. Then, when the prediction time step is $\tau$ , the time series data $z_{i}$ can be expressed as follows,

$$
z _ {i} = f \left(z _ {i - \tau}, c _ {i - \tau}, \varphi\right), \tag {1}
$$

where $c_{i - \tau}$ is a covariate that jointly affects the prediction results with $z_{i - \tau}$ , $f(\cdot)$ represents the nonlinear mapping relationship, $\varphi$ is a random parameter representing the

random fluctuation of the time series.

# III. METHODOLOGY: SPIKE ECHO STATE NETWORK

In this section, in order to solve the above time series prediction problem of aero-engine fault diagnosis, we establish a spike echo state network (Spike-ESN) model composed of spike input layer, spike reservoir and output layer for aero-engine fault prediction.

# A. Spike input layer

The spike input layer can encode the input signal into a spike sequence, which emphasizes the temporal features of data in order to extract critical information.

First, we treat the normalized input data $u(t) \in \mathbb{R}$ as the probability of the spike interval, which determines the average of the spike interval, as follows,

$$
h _ {\kappa} (t) = N _ {s a m} \times \frac {U _ {m a x} - u (t)}{U _ {m a x} - U _ {m i n}}, \tag {2}
$$

where $h_{\kappa}(t) \in \mathbb{R}$ represents the average interval of spike generated by the input data $u(t)$ , $U_{max}$ and $U_{min}$ are the maximum and minimum values of the input time series data, respectively. $N_{sam} \in \mathbb{Z}^{+}$ is the spike sampling times of the spiking neuron. With the increase of spikes sampling times, the input data are projected to a higher time dimension, which is better for extracting effective features.

Then, we use the following Poisson distribution to generate the spike sequence,

$$
P (k) = \frac {m ^ {k}}{k !} e ^ {- m}, \tag {3}
$$

where $k$ represents the number of events, $m$ is the mean and variance of Poisson distribution.

Remark 1. Poisson encoding is one of the effective methods for spike input layers. In theory, The Poisson distribution is suitable for describing the number of random events occurring per unit time, which corresponds exactly to the definition of spike issuance rate. Moreover, Poisson encoding can transform the input image or signal into a spike sequence that conforms to the Poisson process, so that the continuity and sparsity of the input information can be maintained.

Next, all intervals $K(t)$ can be randomly generated with $h_{\kappa}(t)$ as the mean of the Poisson distribution shown in formula (3), i.e.,

$$
K (t) = \left[ \kappa_ {1} (t), \kappa_ {2} (t), \dots , \kappa_ {N _ {\text {i n t}}} (t) \right] ^ {T}, \tag {4}
$$

where for any $l\in \{1,2,\dots ,N_{int}\}$ $\kappa_l(t)\in \{1,2,\dots ,N_{sam}\}$ $N_{int}\in \mathbb{Z}^{+}$ is the number of spike intervals, $N_{sam}$ and $N_{int}$ satisfy the relationship as follows,

$$
\sum_ {l = 1} ^ {N _ {\text {i n t}}} \kappa_ {l} (t) \leq N _ {s a m}. \tag {5}
$$

Finally, the spike sequences are generated by intervals, i.e., according to the following rules,

$$
s _ {i} (t) = \left\{ \begin{array}{l l} 1, & i = \sum_ {j = 1} ^ {k} \kappa_ {j} (t) \\ 0, & \text {o t h e r w i s e} \end{array} , \right. \tag {6}
$$

Figure 3. Spike echo state network. It consists of a spike input layer, a spike reservoir and an output layer.

where $i\in \{1,2,\dots N_{sam}\}$ is the position of the element in the spike sequence, $k\in \{1,2,\dots ,N_{int}\}$ , $j\in \mathbb{Z}^{+}$ .

Based on the above method, the spike input layer $f_{in}(\cdot)$ can convert the input signal into a spike sequence as follows,

$$
f _ {i n} (u (t)) = \left[ s _ {1} (t), s _ {2} (t), \dots , s _ {N _ {s a m}} (t) \right] ^ {T} \in \{0, 1 \} ^ {N _ {s a m}}, \tag {7}
$$

where $[s_1(t), s_2(t), \dots, s_{N_{sam}}(t)]^T$ is the spike sequence converted from the input data, for any $i \in \{1, 2, \dots, N_{sam}\}$ , $s_i(t) \in \{0, 1\}$ , where 1 and 0 represent the activation and inhibition of the spike neuron, respectively.

# B. Spike reservoir

The spike reservoir is a sparse network composed of large-scale neurons randomly connected, which can memorize information by adjusting the internal weights of the network.

First, we generate spike reservoirs in order to represent the input data in a high-dimensional and non-linear manner, as follows,

$$
\boldsymbol {W} _ {r e s} = \rho \cdot \frac {\boldsymbol {W}}{\lambda_ {\max } (\boldsymbol {W})} \in \mathbb {R} ^ {N _ {r e s} \times N _ {r e s}}, \tag {8}
$$

where $\mathbf{W}_{res}$ is the random fixed internal weight of the reservoir, $\mathbf{W} \in \mathbb{R}^{N_{res} \times N_{res}}$ is a matrix with sparse degree $\eta$ randomly generated from a uniform distribution on the interval $[-1, 1]$ . When the data is projected into the high-dimensional and sparse space, the readout of the state will be easier. $N_{res}$ is the number of spike neurons in the spike reservoir, which needs to be set at a moderate level. If this parameter is set too small, the internal features cannot be projected into the high-dimensional space to obtain the best results. On the contrary, if the parameter is set too large, it will cause the introduction of redundant information and difficulties in decoupling state. $\lambda_{\max}(\mathbf{W})$ is the maximum eigenvalue of matrix $\mathbf{W}$ , $\rho$ is the spectral radius that controls and determines the generation of $\mathbf{W}_{res}$ , i.e., the maximum eigenvalue of the adjacency matrix $\mathbf{W}_{res}$ of the reservoir is limited by $\rho$ .

Then, the update mode of the internal state $x(t)\in \mathbb{R}^{N_{res}}$ of the spike reservoir can be expressed as follows,

$$
\left\{ \begin{array}{l} x (t) = \operatorname {t a n h} \left(\boldsymbol {W} _ {\text {i n}} f _ {\text {s p i k e}} \left(t _ {\text {s p i k e}}\right) + \boldsymbol {W} _ {\text {r e s}} x (t - 1)\right), \\ t _ {\text {s p i k e}} = \left[ t _ {\text {s p i k e}} ^ {(1)}, t _ {\text {s p i k e}} ^ {(2)}, \dots , t _ {\text {s p i k e}} ^ {(N _ {\text {i n t}})} \right] ^ {T}, \\ t _ {\text {s p i k e}} ^ {(i)} = \operatorname {P o s i t i o n} \left(s _ {\text {a c t}} ^ {(i)}\right) \in \{1, 2, \dots N _ {\text {s a m}} \}, \\ f _ {\text {s p i k e}} (\cdot) = [ I (1), I (2), \dots , I (N _ {\text {s a m}}) ] ^ {T} \in \mathbb {R} ^ {N _ {\text {s a m}}}, \\ I \left(t _ {\text {s e q}}\right) = \sum_ {i = 1} ^ {N _ {\text {i n t}}} \exp \left(- \frac {t _ {\text {s e q}} - t _ {\text {s p i k e}} ^ {(i)}}{\psi}\right) \in \mathbb {R}, \\ i \in \{1, 2, \dots N _ {\text {i n t}} \}, \end{array} \right. \tag {9}
$$

where $W_{in} \in \mathbb{R}^{N_{res} \times N_{sam}}$ is the input weight matrix of the reservoir, which is generated from the uniform distribution on the interval $[-1, 1]$ , $f_{spike}(\cdot)$ contains the external input current $I(t_{seq})$ of the neuron at each sampling time $t_{seq} = \{1, 2, \dots, N_{sam}\}$ , $t_{spike}^{(i)}$ is the occurrence time of the $i^{th}$ spike, $\psi$ is the time constants of synaptic currents, which controls the overall magnitude level of the current emitted. $s_{act}^{(i)}$ is the $i^{th}$ element with the value of 1 in the spike sequence $[s_1(t), s_2(t), \dots, s_{N_{sam}}(t)]^T$ , i.e., the spike neuron is activated at this moment, and the function $Position(s_{act}^{(i)})$ represents the specific position of the element $s_{act}^{(i)}$ in the spike sequence. In particular, when $t = 1$ , the reservoir is initialized in the empty state, i.e., $x(0) = 0$ .

Remark 2. According to (9), the internal state $x(t)$ is converged when the maximum eigenvalue of $\mathbf{W}_{res}$ is less than 1, i.e., the historical input is gradually forgotten as the network is updated. In short-term time series prediction, the future data are only related to the recent historical data.

Finally, after the reservoir is updated periodically, all internal states $x(t)$ are integrated in the state collection matrix $X$ , as follows,

$$
\boldsymbol {X} = [ x (1), x (2), \dots , x (T) ] \in \mathbb {R} ^ {N _ {\text {r e s}} \times T}, \tag {10}
$$

where $[\cdot, \cdot]$ represents the parallel connection between state vectors, and $T$ is the number of input samples.

# C. Output layer

The output layer $\mathbf{W}_{out} \in \mathbb{R}^{1 \times N_{res}}$ is a weight matrix that needs to be optimized [29], which can parse the internal state of the previous layer into output data, as follows,

$$
\hat {y} = \boldsymbol {W} _ {\text {o u t}} \boldsymbol {X}, \tag {11}
$$

where $\hat{y} = [\hat{y}(1), \hat{y}(2), \dots, \hat{y}(T)] \in \mathbb{R}^{1 \times T}$ is the predicted data.

In order to make the model output data closer to the real value, the optimization problem of the output weight can be constructed as follows,

$$
\min  _ {\boldsymbol {W} _ {o u t}} \| \hat {y} - \boldsymbol {W} _ {o u t} \boldsymbol {X} \| _ {2} ^ {2} + \mu \| \boldsymbol {W} _ {o u t} \| _ {2} ^ {2}, \tag {12}
$$

where $\| \cdot \| _2$ represents the $L_{2}$ norm, $\mu$ is the regularization coefficient, which is used to add the penalty term to the optimization problem, so as to avoid the network from falling into overfitting. The larger the regularization coefficient, the simpler the model and the less risk of overfitting, but it may lead to underfitting. The smaller the regularization coefficient, the more complex the model and the greater the risk of overfitting, but it may improve the performance on the training set.

Generally, the above optimization problem is solved by ridge regression with pseudoinverse, as follows,

$$
\hat {\boldsymbol {W}} _ {\text {o u t}} = y \boldsymbol {X} ^ {T} \left(\boldsymbol {X} \boldsymbol {X} ^ {T} + \mu \boldsymbol {I}\right) ^ {- 1}, \tag {13}
$$

where $\pmb{I}$ is an identity matrix, $y$ is the target output signal.

Above all, the Spike-ESN structure is shown in Figure 3, and the establishment process of the Spike-ESN model is as shown in Algorithm 1. Further, the aircraft fault warning process is shown in Figure 4. Since the scheme in Figure 4 is only one of the parts of the aircraft failure prediction, it is necessary to consider the solution more completely. In practical engineering, the operating environment of aircraft is changeable, so the optimum model parameters need to be adjusted accordingly. The hyperparameter $\psi$ have an impact on the network by keeping the internal state at the right level. Since our dataset is recorded during the actual flight of the aircraft, it contains almost all the operating conditions. Therefore, our network states do not change much as the actual operating conditions change. When the input data produces a large state $x(t)$ , the value of hyperparameter $\psi$ increases in fixed steps. Conversely, when the state $x(t)$ is small, the value of hyperparameter $\psi$ decreases in fixed steps. Finally, the network can start retraining until the internal state value satisfies the stop condition.

Remark 3. Spike-ESN consists of three parts: spike encoding, spatial projection, and readout of internal states. First, the spike encoding part strengthens the feature correlation of sequence with historical data from the perspective of high-dimensional projection in time. Then, the spatial projection divides the features into several easily distinguishable subspaces, which reduces the complexity, coupling and redundancy. Finally, the readout of the internal state part is able to solve the desired law from the adequate features exhibited by the first two parts. Since the training method of Spike-ESN is to solve the least squares loss function by regression, the feature extraction of

Figure 4. Process of using Spike-Esn for aircraft fault early warning.

the network depends largely on whether the features obtained from the front-end of the network are adequate and sparsely separable. Therefore, the spike encoding and spatial projection are proposed in this paper to fully display the temporal features in the high-dimensional sparse space. It is believed that a high-dimensional sparse space can help decoupling, because the sparse projection allows non-zero dense data to be clustered together and form features and patterns for subsequent network extraction. The high-dimensional sparse space can decompose the high-dimensional sparse features into different subspaces, which reduces the complexity and redundancy of the features and improves the extractability and interpretability of the features.

# IV. SIMULATION EXAMPLES

In this section, we present the numerical results of Spike-ESN model for aero-engine fault prediction.

# A. Experimental data and parameter setting

The data we used comes from an aero-engine research institute, which standardized the data in order to comply with the principle of confidentiality. First, we divide the dataset after preprocessing the aero-engine data. The exhaust temperature $T_{6}$ , the combustion chamber temperature $T_{1}$ , high-pressure rotor speed $N_{2}$ , low-pressure rotor speed $N_{1}$ , compressor opening angle $\alpha_{1}$ and $\alpha_{2}$ are selected as prediction targets, where a strong correlation exists between these parameters. In order to make the model learn all the features in the process of aero-engine operation as much as possible, including the normal operation and the faults caused by sudden reduction

# Algorithm 1 Training Algorithm of Spike-ESN.

# Require:

Input signal $u(t)$ , target output signal $y(t)$ , spectral radius $\rho$ , sparsity $\eta$ , regularization coefficient $\mu$ , the spike sampling times $N_{sam}$ , the number of spike intervals $N_{int}$ ;

1: Calculate spike average interval $h_{\kappa}(t) \in \mathbb{R}$ by using formula (2);

2: for $i = 1$ to $N_{int}$ do

3: Generate random intervals $\kappa_{i}(t)$ in Poisson distribution by using formula (3)-(4);

4: end for

5: for $i = 1$ to $N_{sam}$ do

6: Generate spike sequences $s_i(t)$ for each input signal by using formula (6);

7: end for

8: Randomly generate input weight matrix $\mathbf{W}_{in} \in \mathbb{R}^{N_{res} \times N_{sam}}$ in uniform distribution;

9: Calculate the internal weight matrix $\mathbf{W}_{res} \in \mathbb{R}^{N_{res} \times N_{res}}$ with the sparsity of $\eta$ and the spectral radius of $\rho$ by using formula (8);

10: Initialize an empty state collection matrix $X$ ;

11: for $t = 1$ to $T$ do

12: Calculate the internal state $x(t) \in \mathbb{R}^{N_{res}}$ at each time by using formula (9), and add $x(t)$ to the state collection matrix $X$ ;

13: end for

14: Calculate the output weight matrix $\mathbf{W}_{out}$ by using formula (13);

# Ensure:

Output weight matrix $\mathbf{W}_{out} \in \mathbb{R}^{1 \times N_{res}}$ ;

Figure 5. The average spike interval of the exhaust temperature $T_{6}$ and the compressor opening angle $\alpha_{2}$ in the training dataset.

of ambient pressure $D_{8}$ , the experiment takes a period of flight data of the aircraft as the training set, where the fault data is randomly and smoothly inserted into the training set.

Then, the dataset is spiked by using the Poisson distribution coding method in Algorithm 1. For example, when the number of spike samples is set to 100, the average interval of the exhaust temperature $T_{6}$ and the compressor opening angle

Figure 6. Spike encoding result of a period of data of compressor opening angle $\alpha_{2}$ .

$\alpha_{2}$ in the training dataset is shown in Figure 5. In addition, a period of data of compressor opening angle $\alpha_{2}$ are spike encoded as shown in Figure 6, where the frequency of spike sequence generated by large value is high, and the frequency of spike sequence generated by small value is low instead.

In the experiment, we use the proposed Spike-ESN model, the echo state network (ESN) [27], auto regression and moving average (ARMA) [28], convolution neural network (CNN) [30], convolution neural network (LSTM) [31] and Transformer [32] for comparison. In order to make the prediction results as fair as possible, the following parameter settings are designed: the key parameters of the Spike-ESN model and the ESN model are set the same, where the reservoir dimension $N_{res}$ is set to 100, the spectral radius $\rho$ is set to 0.9, the sparsity $\eta$ is set to 0.1, the regularization coefficient $\mu$ of regression calculation is set to $10^{-8}$ . In order to make the echo state within the nonlinear range of $\tanh(\cdot)$ activation function, the scaling factor is set to 0.8. For the spike neurons, the spike sequence length is set to 100, and the time constant $\psi$ of synaptic currents is set to 5000. Before the training, the echo state network needs to be initialized. The first 200 data points in the dataset are used for initialization, which are not included in the network evaluation.

In addition, in order to verify the advantage of the proposed Spike-ESN model in aero-engine fault prediction, ARMA is set as the parameter under optimal conditions. When ARMA model is set to $ARMA(4, 4)$ , the effect of the model reaches the optimal. Actually, the effect of ARMA model increases significantly with the increase of parameters, but further increase of parameters will lead to increased computation rather than better effect. Therefore, AR and MA in ARMA model are set to 4 parameters respectively, i.e., $ARMA(4, 4)$ . Then the structure of CNN is set as 2 COV1D layers-1 MAXPooling1D layer-1 Flatten layer-3 Dense layers, and the sliding window length is set to 15. Next, we set 4 neurons in the LSTM layer and set the number of training cycles to 100. And we set the parameters of Transformer to 4 multi-headed attention layers, 128 feature nodes, 128 fully connected neurons, 2 layers of encoder, 1 layer of decoder, and sliding window is also set to 15. Based on the analysis above, the main settings of the 6 networks are shown in Table I. Finally, Root mean square error (i.e., $RMSE = \sqrt{\sum(\hat{y} - y)^2 / n}$ ) and mean absolute percentage error (i.e., $MAPE = \sum |(\hat{y} - y) / y| / n$ )

are used as quantitative evaluation indicators for verifying the prediction results.

# B. Experimental results

When the time step is 1, the prediction results of the aero-engine parameters by Spike-ESN model are shown in Figure 7, where the fault occurs in the range of 1s to 15s. It can be seen from Figure 7 that the prediction errors of the Spike-ESN model are about $10^{-4}$ , which can accurately reflect the development trend of aero-engine state data in the short term. When the prediction time step increases to 10, the prediction results of the aero-engine parameters by Spike-ESN model are shown in Figure 8, where almost all of the prediction errors are about $10^{-2}$ . Although the prediction error of the Spike-ESN model increases with the increase of time step, it can still predict the development trend of aero-engine states in the future.

Specifically, the six aero-engine state parameters are predicted by six models when the time step varies from 1 to 20 (i.e., as shown in Figure 9), and the corresponding practical physical significance is to predict the changes of aero-engine parameters in the next 0.062s to 1.24s. The prediction errors of the six models with time step of 1, 10 and 20 for each parameter are sorted as shown in Table III.

From Figure 9, it can be noticed that the Spike-ESN model is superior to the ESN model in MAPE. Moreover, with the increase of prediction time step, the convergence of Spike-ESN is better than that of ESN. It can be seen from Table III that Spike-ESN can achieve better performance than ESN in the prediction of all parameters in the experiment. Especially, the accuracy of all parameters has been improved by $2 \text{‰}$ in the long-term prediction with a step size of 20, which may play a key role in aero-engine parameter prediction and fault diagnosis. In a short prediction step, the error of Spike-ESN
model is $10^{-3}$ less than that of ESN model and ARMA model, and with the increase of prediction time step, the advantage of the Spike-ESN model is gradually obvious.

As a traditional mathematical calculation method, ARMA can obtain better results in some prediction steps and has fast training speed, but its convergence and stability are generally worse than those of the other neural networks. ARMA may get very poor results in some prediction steps, so its error curve is not smooth.

In comparison, the RMSE and MAPE of the CNN are almost largest, and the error fluctuation of the CNN is wide. The experimental results show that the CNN possesses a poor ability to extract time series features. The reason is that there is no memory unit with recurrent serial structure in the CNN. Therefore, it is difficult for CNNs to combine a large amount of past time information for analysis.

As shown in Figure 9, the error of LSTM increases with the increase of the prediction step, but always at a low level. As the prediction step increases, LSTM gradually shows the effect of exceeding other models, which is due to the advantage of LSTM's recurrent memory unit for time series information. Comparing the running time, the single run time of LSTM is 99.56s, which is much faster than that of CNN.

From Figure 9, the prediction effect of Transformer is very close to that of CNN, which is caused that both CNN and Transformer are parallel input-output structures without memory units. Transformer relies on positional encoding to capture time-series information which is less effective in time

series. In addition, Transformer is weak in predicting nonstationary time series since it lacks the ability to adapt well to changes in time series data. Further, Transformer's multi-headed attention mechanism is more suitable for large-scale data, such that it is difficult to achieve better results in the small-scale data in this paper. Moreover, the Transformer has a single run time of 155.62s which can also only be used for offline aero-engine fault diagnosis on the ground.

It is important to note that our dataset is a professional dataset with complex properties from aero-engine research institute. The properties of the datasets have a great impact on the prediction accuracy, so it is one-sided to judge the model's performance only from the prediction accuracy. If the predicted time series data are smooth and stable, the RMSE of the poorer prediction results will also exhibit small. In this case, a model with weak predictive ability will exactly match the target if it holds the value of the previous moment. On the contrary, the data of parameter $\alpha_{2}$ in the dataset after the failure are mostly fluctuating, so it is a greater challenge for the prediction algorithm, which can distinguish the performance of the model prediction. For instance, other deep networks such as the LSTM, which are able to extract deep features of time series and have a large number of training parameters, have an advantage in long-time prediction capability. In contrast, our method has the advantage of small parameter size and fast training speed, for which a comparison of the single training time of the model is added. Comparing single run time, SpikeESN takes only 8.31s to complete the training at a spike
sampling count of 50, and only 4.92s to complete the training at a spike sampling count of 20. But all other deep large-scale networks have difficulty to achieve such speed.

As noted above, the data can be spiked by using Spike-ESN model, which can be reflected in the spike space. The Spike-ESN model converts sequence data into spatio-temporal information, which is more conducive for the model to extracting time factors. Then, the temporal and spatial information is converted into the external input current of the ESN neuron by using the spike conversion formula, so that the Spike-ESN model can capture sufficient features. The reason why Spike-ESN is a more stable and accurate model than ESN and ARMA is that Spike-ESN model includes the advantages of small amount of ESN calculation, convenient training and being suitable for time series, meanwhile, the spike mechanism is added to make the new model more sensitive to the time information in the time series. In long-term prediction, it is difficult for the model to establish the input-output mapping relationship for complex signals. Therefore, a higher requirement is put forward for the spatio-temporal feature extraction of the Spike-ESN model. The experimental results show that Spike-ESN has a stronger long-term prediction ability than

ESN, and can eliminate the error accumulation of long-term prediction. Moreover, Spike-ESN does not require iterative optimization, so the training speed is quite fast. Compared with several networks above, Spike-ESN has the characteristics of high prediction accuracy and fast training speed, which is better at small-scale time series prediction.

Based on the above analysis, Table II is presented to list the differences between the six models (CNN, Transformer, LSTM, ARMA, ESN and Spike-ESN) from seven aspects (Bio-interpretability, Spatial-temporal Dynamics, Memory, Sparsity, Stability, Training Speed and Accuracy). As shown in Table II, Spike-ESN has unique bio-interpretability, spatio-temporal dynamics and memory to extract time series features. Besides, it has high performance in terms of stability and accuracy. Due to the sparse structure and the special training method, the training speed of proposed method is high.

To sum up, the Spike-ESN model has enhanced time information feature extraction, which can better learn the features of state instability and rotating stall process before aero-engine surge fault. Furthermore, it can realize the role of short-term early prediction parameters in aero-engine fault diagnosis,

Table III PREDICTION ERRORS (RMSE AND MAPE) FOR PREDICTING 6 PARAMETERS $(T_{6}, T_{1}, N_{2}, N_{1}, \alpha_{1}, \alpha_{2})$ BY 6 MODELS (CNN, TRANSFORMER, LSTM, ARMA, ESN, SPIKE-ESN) AT PREDICTION STEP OF 1, 10, 20.   

which provides favorable conditions for aero-engine to prevent faults in the early stage of surge. The Spike-ESN model has achieved good results on the whole, which reflects the data that will occur in the future of aero-engines more accurately. In this experiment, Spike-ESN model can predict engine parameters more accurately and stably after 1.24s than ESN and ARMA. Significantly, without being affected by sudden changes in the external environment or personnel operation, users can roughly know the possible abnormalities of aero-engine parameters after 1.24s, and then make rapid response. If the aircraft is suddenly affected, the rapid training of the Spike-ESN model can update the latest prediction in real time.

# C. Discussion

In this subsection, we will discuss the benefits of spike encoding and spatial projection, and experimentally explore


Figure 11. The output weight matrices in Spike-ESN and ESN for predicting $\alpha_{2}$ at step of 20.

the effect of spike encoding length on the performance.

1) In order to more fully demonstrate the prediction performance of the proposed method, it is necessary to visualize the deep features of the network. Since the internal state of the reservoir can clearly reflect the deep features in the Spike-ESN, the process of reading out the internal state needs to be visualized. As shown in (11), it is clear to reflect the prediction effect by visualizing internal state matrix $\mathbf{X}$ and output weight $\mathbf{W}_{out}$ in Figures 10 and 11, respectively.

The five curves in Figure 10 represent the time series of the five neuron state in the internal state matrix $\mathbf{X}$ when updated with the input data. Hence, the x-axis in Figure 10 represents the number of input sequence data and the y-axis represents the value of the internal state. In order to compare the effects of the proposed spike encoding method on the deeper features of the network, the first and second subplots in


Figure 12. Value range of the states in the reservoirs of Spike-ESN and ESN for predicting $\alpha_{2}$ at step of 20.

Figure 10 visualize the internal states of Spike-ESN and ESN, respectively. As shown in Figure 10, after spike encoding and neuronal membrane currents for temporal feature extraction, the curves in the first subplot cross each other less, while almost all the curves in the second subplot cross each other a lot, i.e., the input features make the internal states of the network separated and smooth as far as possible.

To explain the benefits of separating and smoothing the state variables, the value range of each state node is plotted in Figure 12. In terms of value range, separating and smoothing increase the value range of the main state nodes and reduce that of the redundant nodes. As shown in Figure 12, the width of the value range between the state nodes in SpikeESN is significantly different, while the value range of the state nodes in ESN is almost the same. As shown in (11), a linear regression method was used to read out the internal state, which can be written as:

$$
\hat {y} = w _ {1} x _ {1} + w _ {2} x _ {2} + \dots + w _ {N _ {r e s}} x _ {N _ {r e s}}, \tag {14}
$$

where, for $i \in \{1, 2, \dots, N_{res}\}$ , $x_i$ represents the internal state, $w_i$ represents the corresponding weight coefficient of each internal state.

The width of the value range of the state node is denoted as $\delta x_{i}$ , and the width of the value range of the predicted result is denoted as $\delta y$ . Then, for each input data corresponding to $x_{i}$ , the value range $\delta x_{i}$ contains both beneficial and harmful components for accurate prediction. If the scale of $\delta x_{i}$ is small, then the corresponding $x_{i}$ is barely changed when the input data is changed. Its contribution to all predictions varies so little that it hardly affects the prediction results. On the contrary, if the scale of $\delta x_{i}$ is large, the corresponding $x_{i}$ changes a lot when the input data is changed, which has a large impact on the prediction results. Further, the state node contains sufficient features, i.e., the beneficial component is large and the harmful component is small, then the prediction results will be more accurate. From Table III, it can be seen

that the RMSE of Spike-ESN is smaller than that of ESN, i.e., the prediction accuracy is better.

Correspondingly, the first and second subplots in Figure 11 visualize the output weights $\mathbf{W}_{out}$ for Spike-ESN and ESN, respectively. In Figure 11, the x-axis and the y-axis indicate ordinal number and value of weights, respectively. For example, the number 25 on the x-axis indicates the $25_{th}$ weight in the output weights. From the comparison experiments, it is found that several output weights in Spike-ESN exhibit large values, while most output weights have low values. Since the output weights correspond to the internal states, it can be seen that Spike-ESN concentrates the redundant components in some internal states, which facilitates to reduce their impact on the prediction effect. On the contrary, the output weights of ESN are uniformly distributed, which will enhance the effect of the redundant components caused during the high-dimensional projection of the features on the output results. In order to clearly distinguish the difference between the output weights $\mathbf{W}_{out}$ in the comparison experiments, a threshold is set to filter out the output weights that significantly affect the network performance. In Figure 11, when the threshold is set to 300, there are 20 and 54 significant output weights for Spike-ESN and ESN, respectively. In addition, when the threshold is set to 200, there are 36 and 70 significant output weights for Spike-ESN and ESN, respectively. When the threshold is set to 400, there are 7 and 46 significant output weights for Spike-ESN and ESN, respectively. Since multiple sets of data illustrate the same law, only one case in Figure 11 is shown, and other settings are omitted.

From Figure 11, it is demonstrated that these 20 weights reflect the time series information more accurately. Further, the value of the beneficial component is large in state variables with large value range width $\delta x_{i}$ . Correspondingly, there are 54 effective weights for ESN, which proved that the information of the time series is not accurately reflected in these states. Meanwhile, the value of the harmful component is large in the states with large value range width $\delta x_{i}$ . Therefore, it can be concluded that spike encoding has a certain enhancement effect on the extraction of state features. To sum up, the states in the first subplot of Figure 10 are more suitable as a base for the linear regression.

Finally, since a large prediction step can better reflect the advantages of the spike encoding approach in temporal feature extraction, we use the sequence with a prediction step of 20 to compare the internal states. The first and second subplots of Figure 10 and Figure 11 represent the internal states and the corresponding output weights of Spike-ESN and ESN for predicting $\alpha_{2}$ at the prediction step of 20, respectively. From Table III, the prediction error RMSE of Spike-ESN is 0.0471 and that of ESN is 0.0536. It can be seen that the proposed method has a significant advantage, which is consistent with the above analysis.

2) Traditionally, the echo state network maps the input signal into a sparse high-dimensional space by a random nonlinear mapping method [29]. Further, the input signal has good spatial features, so that a simple linear method for the output weight training can make the model achieve excellent performance. Therefore, the rationality of this high-

Figure 13. The influence of spike sampling times on model accuracy.

dimensional projection directly affects the quality of the optimization results. However, the echo state network does not have the spatio-temporal dynamics similar to biological neurons, which make it hard to capture the temporal features of time series data. In reality, neurons in biological systems use spikes to capture and transmit signals, which cause neurons to generate membrane potential firing. Typically, we can approximate neuron dynamics as an integral process. In particular, when the neuron membrane potential is higher than a certain threshold voltage, the neuron will emit an action potential to transmit information [33]. In order to transmit different information through action potentials, the size and firing time of the neuron's action potential are critical.

In conclusion, the spiking neuron model can map a physical continuous information flow into a discrete dynamic system in a high-dimensional neural information space, which greatly improves feature extraction and data analysis. Similarly, the high-dimensional projection of spatio-temporal dynamics is more beneficial to the output layer of the echo state network to extract the necessary features.

To verify the effect of spatio-temporal dynamics in Spike-ESN, the test data are collected from actual aero-engine sensors. We adjust the spike sampling frequency to vary the input current sequence length of spike neurons, and then test a signal prediction effect at several time steps separately. Meanwhile, an ESN model without temporal dynamics was used for comparative testing, where the two models share the same key parameters. Since the scales of test results vary greatly, the results are represented by the natural logarithm (i.e., as shown in Figure 13). The parameter $N_{sam}$ is the sampling times of the spiking neuron, i.e., the dimension of the temporal projection. After the triggering of the spike sequence, the membrane potential of the spiking neuron generates a current sequence with the time length of $N_{sam}$ . Then the current sequence is used as the input of the network, which makes the input data projected into a high-dimensional space in time for feature extraction. Figure 13 shows the prediction

results on the dataset by adjusting the value of $N_{sam}$ from 1 to 100. Thus, the effect of $N_{sam}$ (i.e., the temporal projection dimension) on the network performance is obtained. It can be seen from Figure 13 that the higher the temporal projection dimension, the stronger the performance of the network, reflecting the better feature extraction ability. However, after the time dimension is increased to a certain level, the network performance no longer increases, which mean that the features extracted by the projection method are exhausted.

The test results show that the RMSE of Spike-ESN has a significant decrease with the increase of the number of spike sampling, which exceeds the performance of ESN at each prediction step. Based on these results, the introduction of spatio-temporal dynamics and increasing the spike sampling frequency have a positive effect on the prediction results of the model.

# V. CONCLUDING REMARKS

In this work, we designed a spike echo state network (Spike-ESN) model based on the deep learning framework to solve the problem of aero-engine future state prediction. First, the spike input layer based on Poisson distribution in Spike-ESN model can extract useful temporal features from aero-engine sequence data. Then, Spike-ESN model inputs the time characteristics into the spike reservoir through the spike accumulation calculation method, which can project the data into high-dimensional sparse space. Spike-ESN model adopts spike neurodynamics mechanism and time sensitive echo state mechanism, which has the ability to enhance the temporal information characteristics of data and time series memory. Based on the results of aero-engine future state prediction, it is proved that Spike-ESN model is a method with high accuracy and stability. Spike-ESN model can effectively predict the future flight parameters of aero-engines, which lays a good foundation for aero-engine early fault warning.

