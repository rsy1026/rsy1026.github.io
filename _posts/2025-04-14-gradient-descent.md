---
title: "경사하강법 (Gradient Descent) - 처음부터 이해하기"
categories: 
    - deep_learning
tags: 
    - Deep Learning

permalink: deep_learning/gradient_descent

toc: true
toc_sticky: true
layout: single

date: 2025-04-14
last_modified_at: 2026-02-24
---

이 포스트는 **경사하강법(gradient descent)**에 대해 묻는 질문에 스스로 잘 대답할 수 있도록, 공부가 너무 부족했던 기본 개념부터 누적해서 정리한 포스트이다. 개인적으로 검색하며 모아두었던 여러 참고자료들을 취합하고 챗지피티로 검토하며 작성하였다.

*오류가 있다면 댓글로 무엇이든 지적해주시면 정말 감사드리겠습니다.*

### 참고 자료
경사하강법
- [경사하강법(gradient descent) - 공돌이의 수학정리노트 (Angelo's Math Notes)](https://angeloyeo.github.io/2020/08/16/gradient_descent.html)
- [Directional derivatives와 gradient descent method](https://m.blog.naver.com/enewltlr/220912511268)

그래디언트 및 방향도함수
- [Calculus III - Directional Derivatives](https://tutorial.math.lamar.edu/classes/calciii/directionalderiv.aspx)
- [편미분∂ 전미분d 변화량Δ 그래디언트∇](https://velog.io/@seokjin1013/%ED%8E%B8%EB%AF%B8%EB%B6%84%EA%B3%BC-%EC%A0%84%EB%AF%B8%EB%B6%84)
- [그래디언트와 방향도함수(Gradient operator and directional derivative)](https://m.blog.naver.com/cindyvelyn/222147143662)
- [머신러닝을 위한 기초수학 - 다변수함수와 그래디언트](https://velog.io/@zlddp723/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B8%B0%EC%B4%88%EC%88%98%ED%95%99-%EB%8B%A4%EB%B3%80%EC%88%98%ED%95%A8%EC%88%98%EC%99%80-%EA%B7%B8%EB%9E%98%EB%94%94%EC%96%B8%ED%8A%B8)
- [Why the gradient is the direction of steepest ascent](https://youtu.be/TEB2z7ZlRAw?si=I-sAcrIsdlXMJjZJ)
  
수학 개념들
- [Khan Academy - 평균변화율 복습](https://ko.khanacademy.org/math/algebra/x2f8bb11595b61c86:functions/x2f8bb11595b61c86:average-rate-of-change-word-problems/a/average-rate-of-change-review)
- [기본개념 - 도함수의 정의](https://bhsmath.tistory.com/172)
- [벡터의 성분](https://jwmath.tistory.com/490)
- [벡터의 성분과 단위 벡터](https://m.blog.naver.com/seolgoons/222031443313)
- [ML 기초 - 수포자가 이해한 미분과 편미분 (feat. 경사하강법)](https://airsbigdata.tistory.com/191)
- [편미분과 전미분](https://velog.io/@swan9405/%ED%8E%B8%EB%AF%B8%EB%B6%84%EA%B3%BC-%EC%A0%84%EB%AF%B8%EB%B6%84)
- [구면좌표계 속도벡터 쉽게 구하는 방법!](https://post.naver.com/viewer/postView.nhn?volumeNo=29238751&memberNo=28329369)
- [다변수함수의 연쇄법칙(Chain Rule)](https://blog.naver.com/mindo1103/90103548178)

&nbsp;
## 용어 정리 

먼저 그래디언트를 설명할 때 자주 쓰이지만 개인적으로도 많이 혼동했던 "변화" 관련 용어들을 정리한다.
- ``변화율``: **변화의 빠르기** (단위당 변화) &#8594; 변화의 <span style="color:orange">방향과 빠르기</span> 모두 포함
- ``변화량``: 실제로 **얼만큼** 변화했는 지 (총 변화) 
- ``변화율의 크기``: 얼마나 **강하게 혹은 가파르게** 변화하는 지 &#8594; 방향을 제외한 순수한 변화의 <span style="color:orange">빠르기</span>만 의미
- ``변화가 얼마나 큰가``: 문맥 없이 이 표현을 쓰면 수학적으로는 불명확
- ``변화가 얼마나 빠른가``: **변화율** 또는 **변화율의 크기**

&nbsp;
## 개념 정리

이 파트에서는 "그래디언트"를 이해하기 전에 알아야 ~~*알았어야*~~ 하는 수학 개념들을 정리한다.
### 평균변화율, 미분계수, 도함수
<span style="color:orange">평균변화율</span>이란, $y=f(x)$일 때 함수 내 주어진 구간에서 $y$가 $x$에 비해 평균적으로 얼마나 증가하는지, 혹은 한 단위당 얼만큼 증가하는지를 뜻한다. 기하학적으로는, 함수 내 주어진 구간의 양 끝점을 잇는 직선의 기울기이다:

$$
    \frac{\triangle{y}}{\triangle{x}}=\frac{f(b)-f(a)}{b-a}=\frac{f(a+h)-f(a)}{h}
$$

<span style="color:orange">미분계수</span>란, 함수 $y=f(x)$에서 $x=a$일 때 $x$의 증가량이 0에 가까워질 때의 평균변화율을 의미한다. 기하학적으로는, 함수 $y=f(x)$ 상에 있는 점 $(a, f(a))$에서의 접선의 기울기, 즉 **순간변화율**을 의미한다:

$$
    f'(a)=\lim_{\triangle{x}\to0}\frac{\triangle{y}}{\triangle{x}}=\lim_{\triangle{x}\to0}\frac{f(a+\triangle{x})-f(a)}{\triangle{x}}=\lim_{h\to0}\frac{f(a+h)-f(a)}{h}
$$

<span style="color:orange">도함수(derivative)</span>란, 함수 $y=f(x)$가 정의한 구간 중에서 미분 가능한 모든 $x$에 대해 해당 점에서의 미분계수를 대응시키는 **함수**이다. 즉, 각 $x$에서의 순간변화율(기울기)을 구한 값을 모은 함수이다:

$$
    f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}=\frac{dy}{dx}=\frac{d}{dx}f(x)
    $$

### 벡터의 성분과 단위 벡터
<span style="color:orange">벡터</span>는 직교좌표계 상의 각 축에 대한 성분의 합으로 나타낼 수 있다. 만약 원점이 $O$인 2차원 평면좌표가 있을 때, $A(a_1,a_2)$를 종점으로 하는 임의의 벡터 $\overrightarrow{a}$에 대해 다음과 같이 표현할 수 있다:
![image info](/assets/img/vector.png){: width="70%"}
*이 [링크](https://jwmath.tistory.com/490)의 이미지를 재구성했습니다.*

$$
    \begin{aligned}
        \overrightarrow{a}
        =\overrightarrow{OA}
        =\overrightarrow{OA_1}+\overrightarrow{OA_2}
        =a_1\overrightarrow{e_1}+a_2\overrightarrow{e_2}
        =a_1\begin{bmatrix}
            1\\ 
            0
            \end{bmatrix}
        +a_2\begin{bmatrix}
            0\\ 
            1
            \end{bmatrix}
        =[a_1,a_2]
        \end{aligned}
        $$

$$
    \begin{aligned}
        \overrightarrow{e_1}
        =\begin{bmatrix}
            1\\ 
            0
            \end{bmatrix},
        \overrightarrow{e_2}
        =\begin{bmatrix}
            0\\ 
            1
            \end{bmatrix}
        \end{aligned}
        $$
이 때 실수 $a_1, a_2$는 각각 $x$축과 $y$축의 <span style="color:orange">성분</span>이고, $\overrightarrow{e_1},\overrightarrow{e_2}$는 각각 $x$축과 $y$축의 방향을 가지고 두 점 $E_1(0,1)$와 $E_2(1,0)$을 종점으로 하는 크기가 1인 벡터, <span style="color:orange">단위 벡터</span>이다. 즉, 각 축에 대한 단위 벡터의 방향과 각각 $a_1, a_2$의 크기를 가지는 두 벡터 $\overrightarrow{OA_1}, \overrightarrow{OA_2}$의 합이라고 할 수 있다.
$A(a_1,a_2)$를 종점으로 하는 평면 벡터 $\overrightarrow{a}$의 크기는 다음과 같이 피타고라스의 정의에 의해 나타낼 수 있다:

$$
    \vert{\overrightarrow{a}}\vert=\sqrt{(a_1)^2+(a_2)^2}
    $$

위에서는 2차원 좌표계였고, 만약 3차원 좌표계가 있을 때의 임의의 벡터 $\overrightarrow{v}$는 다음과 같이 $x,y,z$축 각각에 대한 성분과 단위 벡터 $\hat{i},\hat{j},\hat{k}$를 이용해 표현할 수 있다:

$$
    \begin{aligned}
        \overrightarrow{v}
        =\begin{bmatrix}
            a\\ 
            b\\ 
            c
            \end{bmatrix}
        =a\begin{bmatrix}
            1\\ 
            0\\ 
            0 
            \end{bmatrix}
        +b\begin{bmatrix}
            0\\ 
            1\\ 
            0
            \end{bmatrix}
        +c\begin{bmatrix}
            0\\ 
            0\\ 
            1
            \end{bmatrix}
        =a\hat{i}+b\hat{j}+c\hat{k}
    \end{aligned}
    $$     
$$
    \begin{aligned}
        \hat{i}
        =\begin{bmatrix}
            1\\ 
            0\\ 
            0
            \end{bmatrix},
        \hat{j}
        =\begin{bmatrix}
            0\\ 
            1\\ 
            0
            \end{bmatrix},
        \hat{k}
        =\begin{bmatrix}
            0\\  
            0\\  
            1
            \end{bmatrix}
    \end{aligned}
    $$
이 때 3차원 벡터 $\overrightarrow{v}$의 크기는 마찬가지로 [피타고라스 정의](https://kukuta.tistory.com/152)를 사용하여 나타낼 수 있다:

$$
    \vert{\overrightarrow{v}}\vert=\sqrt{a^2+b^2+c^2}
    $$

### 편미분, 전미분
<span style="color:orange">편미분(partial derivative)</span>이란, 변수 $x, y$로 이루어진 다변량함수 $f(x,y)$에서 **하나의 변수 $x$ 혹은 $y$만을** 움직였을 때의 함수값의 변화율을 의미한다. 변수 $x$ 혹은 $y$만을 $h$만큼 움직였을 때 각각에 대한 식은 다음과 같다:

$$
    f_x(x,y)=f_x(x_0,y_0)=\frac{\partial{f}}{\partial{x}}(x_0,y_0)=\lim_{h\to0}\frac{f(x_0+h,y_0)-f(x_0,y_0)}{h}$$
$$
    f_y(x,y)=f_x(x_0,y_0)=\frac{\partial{f}}{\partial{y}}(x_0,y_0)=\lim_{h\to0}\frac{f(x_0,y_0+h)-f(x_0,y_0)}{h}$$

위 식에서 $f_x(x,y)$에 대해 $a_0=[x_0,y_0]$을 사용하여 다시 정리하면 다음과 같다:

$$
    f_x(a_0)=\lim_{h\to0}\frac{f(a_0+h\hat{i})-f(a_0)}{h}
    $$

이 때, $\begin{aligned}\hat{i}=\begin{bmatrix}1\\0\end{bmatrix}\end{aligned}$는 $x$축에 대한 단위 벡터이다. \
<span style="color:orange">전미분</span>이란, 다변량함수에서 **모든** 변수들이 움직였을 때에 대한 함수값의 ``변화율``을 의미한다. 모든 변수에 대한 변화율을 종합하여 나타낸 미소 변화량 표현을 **전도함수**(total differential)라고 부르며, $\partial$ 표시 대신 $d$ 표시로서 나타낸다. 만약 다변량함수 $f(x,y,z,...)$가 있을 때 해당 함수의 전도함수는 다음과 같다:

$$
    df(x,y,z,...)=\Big(\frac{\partial{f}}{\partial{x}}\Big)dx+\Big(\frac{\partial{f}}{\partial{y}}\Big)dy+\Big(\frac{\partial{f}}{\partial{z}}\Big)dz+...
    $$

### 연쇄법칙
<span style="color:orange">연쇄법칙(chain rule)</span>이란 합성함수의 미분법이다. 일변수 합성함수 $y=f(g(x))$에 대해 $t=g(x), y=f(t)$로 분해되어 있다면 다음 식이 성립한다:

$$
    \frac{dy}{dx}=\frac{dt}{dx}\frac{dy}{dt}
    $$

이변수 합성함수 $z=f(x,y)$에 대해 $x=g(t), y=h(t)$이고, $f(x,y), g(t), h(t)$ 모두 미분 가능한 함수라면 다음 식이 성립하며, 증명법은 이 [링크](https://blog.naver.com/mindo1103/90103548178)에서 확인할 수 있다:

$$
    \frac{dz}{dt}=\frac{dz}{dx}\frac{dx}{dt}+\frac{dz}{dy}\frac{dy}{dt}
    $$

&nbsp;
## 그래디언트와 방향도함수

경사하강법을 공부하는 과정에서 그래디언트와 방향도함수를 함께 설명하는 글들을 많이 찾을 수 있었는데, 그 이유가 있었다. 
그래디언트와 방향도함수의 관계, 그리고 경사하강법으로 이어지는 맥락을 큰 그림으로 먼저 파악하는 과정이 중요한 것 같다. 크게 정리해보면 다음과 같다:

> - <span style="color:orange">그래디언트</span>는 <span style="color:lightskyblue">함수값이 가장 빠르게 증가하는 방향과 그 증가의 강도</span>를 나타내는 벡터이다.
> - <span style="color:orange">방향도함수</span>는 그래디언트와 방향벡터의 내적으로 이루어져 있고, 각 독립변수가 특정 방향으로 단위 거리만큼 이동했을 때의 함수값의 ``변화율``을 나타낸다. 
> - 다변량함수의 경우 방향도함수의 정의에 따라, 함수값의 변화율이 최대가 되려면 **입력 벡터**의 변화 방향이 그래디언트 벡터와 <span style="color:lightskyblue">같은 방향</span>, 최소가 되려면 그래디언트 벡터와 <span style="color:lightskyblue">반대 방향</span>이어야 한다.
> - <span style="color:orange">경사하강법</span>은 목적함수(cost function)의 ```변화율``` 혹은 **기울기**가 최소인 지점을 찾는 방법으로, **입력 벡터**를 그래디언트와 <span style="color:lightskyblue">반대 방향</span>으로 점진적으로 이동시킨다.  

### 그래디언트 (Gradient)

함수 $y=f(x)$의 도함수를 다시 상기해보자. 독립변수 $x$의 변화량이 0으로 수렴할 때 함수값 $y$의 미소한 ``변화량``(differential)은 다음과 같다:

$$
    dy=f'(x)dx=\Big(\frac{dy}{dx}\Big)dx
    $$

만약 독립변수가 여러 개인 $T=T(x,y,z)$라는 다변량함수가 있다면, $T$의 변화는 세 개의 변수 각각의 변화에 의해 영향을 받을 것이다. 따라서 각 변수의 변화량이 0으로 수렴할 때, $T$의 미소변화량은 일변수함수의 미분과는 달리 편미분을 사용한 전미분의 형태로 나타낼 수 있다:

$$
    dT=\Big(\frac{\partial{T}}{\partial{x}}\Big)dx+\Big(\frac{\partial{T}}{\partial{y}}\Big)dy+\Big(\frac{\partial{T}}{\partial{z}}\Big)dz
    $$

위 식을 좀 더 정리하면, 함수값 $T$의 미소변화량은 각 독립변수에 대한 편미분만으로 이루어진 벡터와 각 독립변수에 대한 미소변화량만으로 이루어진 벡터 간의 내적으로 표현될 수 있다: 

$$
    dT=\Big(\frac{\partial{t}}{\partial{x}}\hat{i}+\frac{\partial{t}}{\partial{y}}\hat{j}+\frac{\partial{t}}{\partial{z}}\hat{k}\Big)\cdot(dx\hat{i}+dy\hat{j}+dz\hat{k})=\nabla{T}\cdot d\vec{s}
    $$

위 식에서 편미분으로 이루어진 벡터가 **그래디언트**(gradient) $\nabla{T}$ 이고, 미소변화량으로 이루어진 벡터가 **미소변위벡터** $d\vec{s}$이다. <span style="color:orange">그래디언트</span>는 각 독립변수의 ``변화율``로 이루어진 벡터로, 함수값이 <span style="color:lightskyblue">"어떤" 방향으로 얼마나 빠르게 변화하는 지</span>를 나타낸다. <span style="color:orange">미소변위벡터</span>란, 직교좌표계에서 어떤 순간에 어느 한 지점 $(x,y,z)$에서 다른 지점 $(x+dx, y+dy, z+dz)$으로 <span style="color:lightskyblue">아주 짧게 이동할 때의 한 방향과 미소한 거리</span>에 대한 벡터이다.

정리하면, *함수값의 미소한 변화량은, 1) 함수값이 얼마나 빠르게 변화하는지를 나타내는 **그래디언트**와 2) 독립변수(입력) 벡터의 이동 거리와 방향을 나타내는 **미소변위벡터**의 내적으로 표현될 수 있다*. 이 때, 미소변위벡터 자체는 특정한 방향의 미소 이동만을 나타내지만, 만약 이를 임의의 방향으로 설정할 수 있다면 이는 모든 독립변수의 가능한 방향, 즉 **입력 공간 전체의 방향**을 일반적으로 포함하는 개념이 된다.
따라서 이 내적 표현은 <span style="color:lightskyblue">임의의 방향과 거리에서의 함수값의 변화율</span>을 기술하는 수식이 되어, 함수가 입력값들의 모든 방향 변화에 어떻게 반응하는지, 즉 함수의 **전방위 반응 구조**를 나타낸다고 볼 수 있다.

위 식으로부터 비롯된 다변수 그래디언트 $\nabla{f}$와 해당 그래디언트의 크기 $\Vert\nabla{f}\Vert$를 정의하면 아래 식과 같다:

$$
    \nabla{f}=(\frac{\partial{f}}{\partial{x_1}},\frac{\partial{f}}{\partial{x_2}},...,\frac{\partial{f}}{\partial{x_n}})=\frac{\partial{f}}{\partial{x}}=\nabla_x{f(x)}
    $$
$$
    \Vert\nabla{f}\Vert=\sqrt{\Big(\frac{\partial{f}}{\partial{x_1}}\Big)^2+\Big(\frac{\partial{f}}{\partial{x_2}}\Big)^2+...+\Big(\frac{\partial{f}}{\partial{x_n}}\Big)^2}
    $$

<span style="color:orange">그래디언트 벡터의 **크기**</span>는 모든 독립변수에 대한 ``변화율의 크기``를 의미한다. 즉, 모든 독립변수에 대해 함수값이 얼마나 빠르게 변화하는 지에 대한 증감의 강도를 나타낸다. 여기에서 용어적으로 헷갈릴 수 있는데, **변화의 방향이 무시**된 절대적인 변화의 강도(세기)라고 생각하면 된다. 이 크기가 가장 큰 방향으로 이동하면 함수값이 가장 크게 증가하며, 뒤에 설명할 **방향도함수**의 정의에 의해, 그래디언트는 <span style="color:lightskyblue">함수값이 가장 크게 증가하는 방향과 그 증가의 강도</span>를 나타낸다고 할 수 있다.

### 방향도함수 (Directional Derivative)
위에서 함수값 $T$의 미소변화량은 **그래디언트**와 **미소변위벡터**의 내적으로 표현될 수 있다고 하였다. 이 때, 임의의 방향 대신 "하나의 특정 방향"과 미소한 거리를 "단위 거리"로 제한하였을 때의 변화율을 보는 것이 바로 **방향도함수**(directional derivative)이다. 다시 말해, <span style="color:orange">방향도함수</span>란 입력값을 **특정 방향**으로 **단위 거리**만큼 움직였을 때의 함수값의 변화율을 의미한다. 따라서 방향도함수는 그래디언트와 거리가 포함되지 않은 **방향벡터**와의 내적으로 표현되며, <span style="color:lightskyblue">그래디언트가 특정 방향에 대해 얼마나 투영(projection)되는 지</span>를 보여준다:

$$
    \frac{dT}{ds}=\nabla{F}\cdot \vec{v}
    $$

이 때 $ds$는 스칼라 거리를 의미한다. 즉, 함수값의 미소변화량을 스칼라 거리로 미분했을 때, 이는 거리와 상관없이 계산되는 특정 방향에 대한 변화율을 보는 것과 같으며, 따라서 <span style="color:orange">단위 거리당 변화율</span>을 보는 것이라고 할 수 있다. 
### 방향도함수의 수학적 정의
(이 [링크](https://tutorial.math.lamar.edu/classes/calciii/directionalderiv.aspx)의 글을 번역하여 정리하였다.)

함수 $f(x,y)$에서, 입력값이 $\overrightarrow{u}=\langle{a,b}\rangle$으로 변화할 때 함수값의 변화율을 <span style="color:orange">방향도함수</span>라고 정의하며, $D_{\overrightarrow{u}}f(x,y)$로 표기한다:

$$
    D_{\overrightarrow{u}}f(x,y)=\lim\limits_{h\to0}(\frac{f(x+ah,y+bh)-f(x,y)}{h})
    $$

입력값이 변화하는 위 함수의 식을 더 간단하게 유도하면 다음과 같다:

$$
    g(z)=f(x_0+az,y_0+bz)
    $$

여기에서 $x_0,y_0,a,b$는 어떤 고정된 상수라고 가정하고, $z$만이 유일한 변수라고 해보자. 이 때의 직관은, 고정된 시작점 $(x_0,y_0)$으로부터 고정된 $\langle{a,b}\rangle$ 방향벡터와 방향이 같은 임의의 방향벡터 $\langle{az,bz}\rangle$만큼 $g(z)$가 이동한 지점이라고 볼 수 있다. 즉, $z$가 무엇인지에 따라서 $\langle{a,b}\rangle$ 방향벡터와 같은 방향을 나타내는 방향벡터들이 다양하게 표현될 수 있다.

$g(z)$를 $z$에 대해 미분하면, 미분의 기본 정의(극한을 이용한 변화율)에 따라 다음과 같이 표현할 수 있다:

$$
    g'(z)=\lim\limits_{h\to0}\frac{g(z+h)-g(z)}{h}
    $$

여기에서 $h$는 $z$의 변화량을 나타내는 변수이다. $z=0$일 때 미분계수는 다음과 같다:

$$
    g'(0)=\lim\limits_{h\to0}\frac{g(h)-g(0)}{h}
    $$

아까 위에서의 식 $g(z)=f(x_0+az,y_0+bz)$에서 $z$ 자리에 $h$ 혹은 0을 넣어서 위 식을 치환하면 다음과 같다:

$$
    g'(0)=\lim\limits_{h\to0}\frac{g(h)-g(0)}{h}=\lim\limits_{h\to0}(\frac{f(x_0+ah,y_0+bh)-f(x_0,y_0)}{h})=D_{\overrightarrow{u}}f(x_0,y_0)
    $$
$$
    g'(0)=D_{\overrightarrow{u}}f(x_0,y_0)
    $$

이번에는 다시 같은 $g(z)$ 식을 좀 더 간단하게 표현해보자:

$$
    g(z)=f(x,y)\quad\text{where}\hspace{0.5em}x=x_0+az\hspace{0.5em}\text{and}\hspace{0.5em}y=y_0+bz
    $$

연쇄법칙(chain rule)을 사용해서 위 식을 미분해보면 다음과 같다:

$$
    g'(z)=\frac{dg}{dz}=\frac{\partial{f}}{\partial{x}}\frac{dx}{dz}+\frac{\partial{f}}{\partial{y}}\frac{dy}{dz}
    $$

$x=x_0+az$이므로, $x$를 $z$로 미분하면 $a$, 마찬가지로 $y$를 $z$로 미분하면 $b$가 될 수 있다. 따라서, $x$ 및 $y$에 대한 편미분을 사용해서 아래와 같이 정리할 수 있다:

$$
    g'(z)=f_x(x,y)a+f_y(x,y)b
    $$

$g(0)=f(x_0,y_0)$이므로, 이 식을 미분한 뒤에 일반적인 식으로 다시 표현하면 차례대로 다음과 같다:

$$
    g'(0)=f_x(x_0,y_0)a+f_y(x_0,y_0)b=D_{\overrightarrow{u}}f(x_0,y_0)
    $$
$$
    D_{\overrightarrow{u}}f(x,y)=f_x(x,y)a+f_y(x,y)b
    $$
$$
    D_{\overrightarrow{u}}f(x,y)=\langle{f_x,f_y}\rangle\cdot\langle{a,b}\rangle
    $$
$$
    \nabla f=\langle{f_x,f_y}\rangle\quad\overrightarrow{u}=\langle{a,b}\rangle
    $$
$$
    D_{\overrightarrow{u}}f(x,y)=D_{\overrightarrow{u}}f=\nabla f\cdot \overrightarrow{u}
    $$

$g(z)$는 입력값을 $\vec{u}$ 방향으로 이동시키는 함수이므로, $g'(0)$은 $z=0$일 때 입력값이 $\vec{u}$ 방향으로 '미소하게' 이동했을 때의 변화율을 의미한다. 이는 방향도함수 $D_{\vec{u}}f(x_0,y_0)$의 정의와 일치한다.

결과적으로, 방향도함수 $D_{\overrightarrow{u}}f$는, 다변량함수 $f$의 그래디언트(``변화율`` 벡터) $\nabla f$와 임의의 방향벡터 $\overrightarrow{u}$ 간의 내적으로 표현될 수 있다: 

$$
    D_{\overrightarrow{u}}f(x,y)=D_{\overrightarrow{u}}f=\nabla f\cdot \overrightarrow{u}=\Vert\nabla f\Vert\Vert\overrightarrow{u}\Vert cos\theta=\Vert\nabla f\Vert cos\theta
    $$

이 때 $\theta$는 <span style="color:orange">다변수 그래디언트 벡터와 방향벡터 사이의 사잇각</span>이다. $D_{\overrightarrow{u}}f$가 $\theta$는 <span style="color:lightskyblue">최대값</span>이 되려면 $cos\theta=1$이 되어야 하므로 $\theta=0$ (두 벡터가 <span style="color:lightskyblue">같은</span> 방향)이 되어야 하고, 반대로 $D_{\overrightarrow{u}}f$가 <span style="color:lightskyblue">최소값</span>이 되려면 $cos\theta=-1$이 되어야 하므로 $\theta=180$ (두 벡터가 <span style="color:lightskyblue">반대</span> 방향)이 되어야 한다. 

따라서, 그래디언트 벡터는 그 자체로 함수값의 특정 방향으로의 변화율이 최대가 되는, 즉 <span style="color:lightskyblue">함수값이 가장 빠르게 변하는 방향과 크기</span>를 나타낸다고 할 수 있다. 반대로, **함수값의 특정 방향으로의 변화율이 최소가 되려면 입력 벡터가 그래디언트와 반대 방향으로 이동해야 함**을 알 수 있다.

&nbsp;
## 경사하강법 (Gradient Descent)

드디어 경사하강법이다!

방향도함수의 수학적 정의로부터, <span style="color:orange">다변량함수의 함수값의 변화율이 최소가 되기 위해서는 다변수 입력벡터가 다변수 그래디언트 벡터의 방향과 **정반대** 방향으로 이동해야 한다는 것</span>까지 살펴보았다.  

딥러닝 모델을 학습할 때 우리는 **cost function** 혹은 loss function이라고 불리는 목적함수의 결과값을 계산한다. 이 결과값, 한마디로 예측값과 정답값 사이의 오차에 해당되는 이 값이 최소가 되어서 모델이 우리가 원하는 정답을 잘 학습하면 좋겠다. \
그럼 깔끔하게 목적함수의 미분계수 혹은 기울기(다변수 그래디언트의 모든 성분)가 0인 지점을 찾으면 되지 않을까? \
입력 벡터 $\mathrm{x}=[x_1, x_2, ..., x_N]$에 대한 다변량 목적함수 $y=f(\mathrm{x})$이 있을 때 다음과 같은 비선형 연립 방정식을 풀어야 한다: 

$$
    \nabla f(\mathrm{x})=0
    $$
$$
    \frac{\partial{f}}{\partial{x_1}}=0, \frac{\partial{f}}{\partial{x_2}}=0, ..., \frac{\partial{f}}{\partial{x_N}}=0
    $$


하지만 현실적으로 아무리 실제 문제를 단순화하더라도, 목적함수 $f$ 내부에는 수많은 가중치와 다양한 비선형 함수들이 조합되어 있다. 이로 인해 목적함수를 해가 유한한 닫힌 수식 형태로 풀어내기에는 매우 어렵고 복잡한 문제가 된다. 따라서 직접 모든 가능한 미분계수를 계산하는 것은 현실적으로 쉽지 않을 뿐더러, 최대한 시도한다 하더라도 그 계산량이 어마무시할 것이다. 

이에 경사하강법에서는 gradient가 이미 <span style="color:orange">함수값이 가장 빠르게 변화하는 방향</span>을 나타내므로 목적함수의 그래디언트가 0이 되는 지점(critical point)을 효율적으로 찾아나갈 수 있다. 이 지점은 목적함수 내 여러 개의 local minimum나 saddle point일 수 있고 또는 global minimum일 수 있다. 경사하강법은 즉 주어진 조건에서 **local minimum**을 찾는 방법이며, *global minimum에 도달할지는 문제 특성에 따라 달라진다*. 또한 경사하강법은 같은 과정을 순환적으로(iteratively) 되풀이하며 목적 지점을 점진적으로 찾기 때문에 연산 과정을 최적화하는 데 효과적일 수 있다.   

만약 cost function이 독립변수가 하나인 2차 함수일 때, 다음 그림과 같은 형태로 경사하강법을 시행할 수 있다. 
![image info](/assets/img/gradient_descent.png){: width="60%"}
*이 [링크](https://angeloyeo.github.io/2020/08/16/gradient_descent.html) 내 이미지를 재구성했습니다.*

함수값의 기울기 혹은 그래디언트 $\frac{dy}{dx}$가 양수일 때는 그래디언트의 방향이 양의 방향($+$)이므로 입력값이 그 반대 방향인 음의 방향($-$)으로 이동해야 하고, 반대로 그래디언트가 음수일 때는 입력값이 그 반대 방향인 양의 방향($+$)으로 이동해야 한다. 이를 수식으로 정리하면, 일반적으로 경사하강법을 공부할 때 많이 볼 수 있는 다음과 같은 형태가 된다:

$$
    x_{t+1}=x_{t}-\epsilon\nabla f(x_{t})
    $$

이 때 $\epsilon$은 임의의 학습률(learning rate)로, 이 학습률을 어떻게 잘 설정하느냐에 따라 입력값이 함수의 기울기가 최소인 지점까지 점진적으로 매끄럽게 잘 수렴될 수 있는지가 결정된다.

독립변수가 여러 개인 다변량함수에서는, 이론적으로 **입력 벡터** 전체가 **그래디언트 벡터**의 방향과 반대 방향으로 이동해야한다. 위에서 정의했던 입력 벡터 $\mathrm{x}=[x_1, x_2, ..., x_N]$에 대해서 경사하강법 식을 바꾸면 다음과 같다:

$$
    \mathrm{x}^{t+1}=\mathrm{x}^{t}-\epsilon\nabla f(\mathrm{x}^{t})
    $$

다시 상기하자면, 우리는 다변량 목적함수의 기울기가 최소인 지점을 찾는 것이다. 따라서 아래와 같이 다변수 그래디언트 벡터의 모든 요소가 최소인 지점을 찾는 것이라고 할 수 있다:   

$$
    \nabla f(\mathrm{x})\approx0
    $$
$$
    \frac{\partial{f}}{\partial{x_1}}\approx0, \frac{\partial{f}}{\partial{x_2}}\approx0, ..., \frac{\partial{f}}{\partial{x_N}}\approx0
    $$

딥러닝 모델 학습에서 목적함수의 실질적인 입력 변수들은 <span style="color:orange">모델을 구성하는 파라미터 혹은 가중치(weight)</span>이다. 모델의 입출력 데이터는 학습 과정에서 변하지 않는 **상수**이고, 모델 파라미터에서 값의 변동이 일어나기 때문이다. 각 가중치는 많은 경우 다차원으로 구성되어 있으며, 수학적으로 가중치 벡터 전체가 그의 다변수 그래디언트 벡터와 반대 방향으로 이동해야 한다. 하지만 실제로 구현할 때는 입력 벡터가 움직여야 하는 방향을 각 차원으로 분해 혹은 **투영**(projection)하여, 각 차원에 대해 따로 그래디언트를 구해 업데이트할 수 있다. "투영"한다는 것은, 그래디언트 벡터의 요소가 각각 **편미분**이기 때문에 그 요소 자체가 각 차원으로의 투영이라고 할 수 있다. 즉, 각 차원을 독립적인 방향으로 움직임으로서 전체 벡터가 움직이는 것이라고 볼 수 있는 것이다. 

모델 가중치 $\mathrm{w}=[w_1,w_2,...,w_N]$가 있을 때, 위 식을 $w$에 대해 다시 정리하면 다음과 같다:

$$
    \mathrm{w}^{t+1}=\mathrm{w}^{t}-\epsilon\nabla f(\mathrm{w}^{t})
    $$
$$
    w^{t+1}_n=w^{t}_n-\epsilon\nabla f(w^{t})
    $$

위 첫 번째 식은 가중치 벡터 전체의 이동 방향에 대해 업데이트하는 식이고, 두 번째 식은 가중치의 각 차원에 대해 이동 방향을 투영하여 업데이트하는 식이다. 

&nbsp;
## 정리

- 함수값의 ``변화율``은 그래디언트 벡터와 특정 방향 벡터의 내적, 즉 <span style="color:orange">방향도함수(directional derivative)</span>로 표현된다.
  - 방향도함수 $D_{\overrightarrow{u}}f(x,y)$의 정의: 다변량함수 $f(x,y)$ 위의 점 $P_0=(x,y)$이 임의의 $\overrightarrow{u}$ 방향으로 움직일 때의 ``변화율``
  - <span style="color:orange">함수값이 어떤 방향으로 얼마나 변화하는 지</span>를 나타내며, 이는 각 독립변수에 대한 변화율과 변화 방향의 조합에 따라 결정된다. 
- 우리가 구하고 싶은 목적함수 $f(x,y)$의 방향도함수, 혹은 목적함수값의 ``변화율``이 **최소**가 되어야 변화율이 0인 critical point에 가까워지므로 우리의 목적을 달성할 수 있다.
- 이 때 내적의 정의에 따라, $\overrightarrow{u}$ 방향 벡터가 다변량함수의 그래디언트 벡터 $\nabla f$와 <span style="color:orange">반대 방향</span>이어야 목적함수의 변화율이 최소가 될 수 있다.
- 다시 말해, 입력 벡터가 그래디언트 벡터 $\nabla f$와 반대 방향(음의 부호)으로 움직여야 목적함수를 최소화할 수 있다. (양의 기울기 → 음의 방향 / 음의 기울기 → 양의 방향)
- 실제 경사하강법을 구현할 때에는, 그래디언트 벡터의 각 요소가 **편미분**이므로, <span style="color:orange">입력 벡터 전체의 이동 방향을 각 차원에 대해 투영</span>하여 각 입력 차원을 **독립적**으로 업데이트한다.

&nbsp;
&nbsp;
<script src="https://utteranc.es/client.js"
        repo="rsy1026/rsy1026.github.io-comments"
        issue-term="pathname"
        theme="github-light"
        crossorigin="anonymous"
        async>
</script>