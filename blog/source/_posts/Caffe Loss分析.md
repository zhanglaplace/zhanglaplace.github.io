---
title: Caffe Loss分析
date: 2017-10-20 11:16:01
tags: [Caffe,DeepLearning]
categories: Caffe
---

### Caffe_Loss
  损失函数为深度学习中重要的一个组成部分，各种优化算法均是基于Loss来的，损失函数的设计好坏很大程度下能够影响最终网络学习的好坏。派生于 $LossLayer$,根据不同的Loss层有不同的参数;
#### 1.基本函数
    主要包含构造函数，前向、后向以及Reshape，部分有SetUp的函数，每层都有Loss参数
```cpp
    explicit XXXLossLayer(const LayerParameter& param):
    LossLayer<Dtype>(param),diff_() {}
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
```
<!--more-->

#### 2.常用损失函数
    由于训练中，采用mini_batch的模式
##### (1) EuclideanLoss (欧式损失函数，L2损失)
  $EuclideanLoss$的公式表达为 $loss = \frac{1}{2n}\sum_{i=1}^n{(y_{i}-\hat{y}_{i})^2}$
```cpp
  //reshape函数，完成层次的reshape,diff_与输入的N*C维度相同
  template <typename Dtype>
  void EuclideanLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
      LossLayer<Dtype>::Reshape(bottom,top);//先调用基类的Reshape函数
      CHECK_EQ(bottom[0]->count(1),bottom[1]->count(1));//label类别
      diff_.Reshape(*bottom[0]);//一般是N*C*1*1
  }

  // Forward_cpu 前向 主要计算loss
  template <typename Dtype>
  void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
     const int count = bottom[0]->count();
     caffe_sub(count,
               bottom[0]->cpu_data(),//网络的输出 N*C
               bottom[1]->cpu_data(),//对应label N*C
               diff_.mutable_cpu_data()//对应的loss差分
           );//完成 y_{predicy}-y_{label} //bottom[0]-bottom[1]
     Dtype dot = caffe_cpu_dot(count,diff_.cpu_data(),diff_.cpu_data());
     //bottom[0]->num()== bottom[0].shape(0);
     Dtype loss = dot/bottom[0]->num()/Dtype(2);//loss/(2*n)
     top[0]->mutable_cpu_data()[0] = loss;
  }

 //Backward_cpu f'(x) = 1/n*(y_{predict}-y_{label})
 template <typename Dtype>
 void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>&propagate_down,const vector<Blob<Dtype>*>& bottom){
    for (size_t i = 0; i < 2; i++) {
        if (propagate_down[i]) {//需要backward
            //对应predict-label 如果label为bottom[0]就需要乘以-1
            const Dtype sign = (i==0) ? 1 : -1;
            //top[0]->cpu_diff()返回float* length = 1;下式为loss/n;
            const Dtype alpha = sign*top[0]->cpu_diff()[0]/bottom[0]->num();
            //y = ax+by ;
            caffe_cpu_axpby(bottom[0]->count(),//count
                            alpha,// loss/n
                            diff_.cpu_data(),//y_{predict}-y_{label}
                            Dtype(0),
                            bottom[i]->mutable_cpu_diff()
                        );//1/n*loss*(y_{predict}-y_{label})
        }
    }
    //欧式损失函数形式简单，常用于做回归分析，做分类需要统一量纲。
 }
```

##### (2)SoftmaxWithLoss Softmax损失函数
$\qquad softmax函数将输出的各个类别的概率值进行归一化，生成各个类别的prob$
$\qquad 常用的分类损失函数，Softmax输出与Multinomial Logistic Loss的结合。公式如下:$
$$ y_i = softmax(x_i) = \frac{exp(x_i)}{\sum_{j=1}^{n}{exp(x_j)}}$$
$$loss = -log(y_k) ,k为实际的样本label$$
$\qquad 损失函数的推导:\frac{\partial Loss}{\partial x_i}=\sum_{j=1}^{n}{\frac{\partial loss}{\partial y_j}\*\frac{\partial y_j}{\partial x_i}}=-\frac{1}{y_k}\*\frac{\partial y_k}{\partial x_i} \quad k为实际的label,其他的\frac{\partial loss}{\partial y_j} =0 \\$
$$
\qquad \frac{\partial y_k}{\partial x_i} = \frac{\partial softmax(x_k)}{\partial x_i}=
\begin{cases}
\  y_k\*(1-y_k) \qquad k == i \\\
\\
\ -y_k*y_i \qquad \qquad k \,\,!=\,i
\end{cases}
$$
$$
整理后可以发现\frac{\partial loss}{\partial x_i}=
\begin{cases}
\  y_k-1 \qquad k \,== \,i ，即i为实际label\\\
\\
\  y_i \qquad \qquad k \,\,!=\,i,即i不是实际label
\end{cases}
$$
    具体代码的实现如下所示:
1.SoftmaxWithLossLayer的输入:bottom
```cpp
    // bottom[0]为前层的特征输出，一般维度为N*C*1*1
    // bottom[1]为来自data层的样本标签，一般维度为N*1*1*1;
    // 申明
    const vector<Blob<Dtype>*>& bottom;
    //backward部分代码
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();//label
```
2.SoftmaxWithLossLayer层的输出:top
```cpp
    // SoftmaxWithLossLayer的输出其实就是1*1*1*1的最终loss
    // 如果有多个的话实际就是也会保存softmax的输出，但是需要注意的是内部包含了
    //Softmax的FORWAR过程，产生的概率值保存在prob_内
    const vector<Blob<Dtype>*>& top;
    //forward部分代码 ,
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
    if (top.size() == 2) {
        top[1]->ShareData(prob_);//top[1]保存softmax的前向概率
    }
```
3.SoftmaxWithLossLayer的关键变量: $softmax\_top\_vec\_,prob\_$ 记录中间值
```cpp
    shared_ptr<Layer<Dtype> > softmax_layer_;
    /// prob stores the output probability predictions from the SoftmaxLayer.
    Blob<Dtype> prob_;
    /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
    vector<Blob<Dtype>*> softmax_bottom_vec_;
    /// top vector holder used in call to the underlying SoftmaxLayer::Forward
    vector<Blob<Dtype>*> softmax_top_vec_;
    /// Whether to ignore instances with a certain label.
    bool has_ignore_label_;
    /// The label indicating that an instance should be ignored.
    int ignore_label_;
    /// How to normalize the output loss.
    LossParameter_NormalizationMode normalization_;

    int softmax_axis_, outer_num_, inner_num_;//softmax的输出与Loss的维度
    template <typename Dtype>
    void SoftmaxWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
        LossLayer<Dtype>::Reshape(bottom,top);//先调用基类的reshape
        softmax_layer_->Reshape(softmax_bottom_vec,softmax_top_vec_);
        int axis = this->layer_param_.softmax_param().axis();//softmaxproto参数(1)
        softmax_axis_ = bottom[0]->CanonicalAxisIndex(axis);//正不变负倒数
        outer_num_ = bottom[0]->count(0,softmax_axis_);// N mini_batch_size
        inner_num_ = bottom[0]->count(softmax_axis_+1);// H*W 一般为1*1
        //保证outer_num_*inner_num_ = bottom[1]->count();//bottom[1]为label N
        if (top.size() >= 2) {//多个top实际上是并列的，prob_值完全一致
            top[1]->Reshapelike(*bottom[0]);
        }
    }

    //forward是一个计算loss的过程，loss为-log(p_label)
    //由于softmaxWithLoss包含了Softmax所以需要经过Softmax的前向，并得到每个类别概率值
    template <typename Dtype>
    void SoftmaxWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
        //调用Softmax的前向
        softmax_layer_->Forward(softmax_bottom_vec_,softmax_top_vec_);
        //这里等同于softmax_top_vec_[0]->cpu_data();
        const Dtype* prob_data = prob_.cpu_data();
        const Dtype* label = bottom[1]->cpu_data();//label 一般来自Data层
        // 一般是N*C(n个样本，每个C个预测概率)/ N == 类别数目
        int dim = prob_.count()/out_num_;
        int count = 0;//统计实际参与loss的样本个数
        Dtype loss = 0;
        for (size_t i = 0; i < outer_num_; i++) {//每个样本遍历
            for (size_t j = 0; j < inner_num_; j++) { //可以认为j == 0 绝大多数成立
                const int label_value = static_cast<int>(label[i*inner_num_+j]);
                if(has_ignore_label_ && label_value == ignore_label_){
                    // softmaxLayer的参数，可以选择不参与loss的类别
                    continue;
                }
                else{//实际需要判断label_value > 0 ,< prob_.shape(1)
                    // -= 因为loss = -log(p_label),prob_data 是n*c的
                    loss -= log(std::max(prob_data[i*dim+label_value*inner_num_+j)],
                                    Dtype(FLT_MIN)));//防止溢出或prob出现NAN
                    ++count;
                }
            }
        }
        //全部样本遍历完成后，可以进行归一，其实也挺简单，
        // top[0]->mutable_cpu_data[0] = loss/归一化
    }

    // Backward_cpu,这里的Backward实际需要更新的是softmax的输入接口的数据，
    // 中间有个y的转化，具体公式上面已经写出
    // bottom_diff = top_diff * softmaxWithloss' = top_diff * {p -1 或者 p}
    template <typename Dtype>
    void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
        //fc输出与label的位置固定了，因此不需要如同欧式loss去判断label和fc的输入位置
        if (propagate_down[1]) {
            //label不需要backpropagate
        }
        if (propagate_down[0]) {//输入，需要更新
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();//需要修改的
            const Dtype* prob_data = prob_.cpu_data();//N*C
            //这里把diff先确定为softmax输出的y值，即bottom_diff[t] = y_t ;
            caffe_copy(prob_.count(),prob_data,bottom_diff);
            const Dtype* label = bottom[1]->cpu_data();
            // 也可以替换为bottom[1]->count(),实际就是类别C
            int dim = prob_.count()/ outer_num_;//NC/C == N
            int count = 0;
            for (size_t i = 0; i < outer_num_; i++) { //n个样本
                for (size_t j = 0; j < inner_num_; j++) { // 实际j == 0
                    const int label_value = static_cast<int>(label[i*inner_num_+j]);
                    if (has_ignore_label_ && label_value == ignore_label_) {
                        //正好是忽略loss的类别
                        bottom_diff[i*dim+label_vale*inner_num_+j] = 0;
                    }
                    else{
                        //这里需要考虑为什么，实际上之前所有的diff初始为y_t，
                        //根据softmax的偏导知道真实label是y_t -1;
                        bottom_diff[i*dim+label_vale*inner_num_+j] -= 1;
                        ++count;
                    }
                }
            }
            //这里只完成了loss的一部分，还差top_diff即Loss
            //如果归一化，就进行归一，同cpu_forward
            //cpu_diff可以认为是Loss
            // Dtype loss_weight = top[0]->cpu_diff()[0]/归一化
            caffe_scal(prob_count(),loss_weight,bottom_diff);
        }
    }

```
>本文作者： 张峰
>本文链接：[https://zhanglaplace.github.io/2017/10/20]( https://zhanglaplace.github.io/2017/10/20/Caffe%20Loss%E5%88%86%E6%9E%90/)
>版权声明： 本博客所有文章，均采用 CC BY-NC-SA 3.0 许可协议。转载请注明出处！
