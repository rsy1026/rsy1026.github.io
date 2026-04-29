---
title: "[처음부터 이해하기] 경사하강법 (Gradient Descent) (2) - 미분과 연쇄법칙"
categories: 
    - deep_learning
tags: 
    - machine learning
    - deep learning
    - derivative
    - chain rule

permalink: deep_learning/gradient_descent_derivative

toc: true
toc_sticky: true
layout: single
sitemap:
  changefreq: daily
  priority : 1.0

date: 2026-04-28
last_modified_at: 2026-04-28
---

경사하강법의 "경사"를 배우기 위해 꼭 필요한 개념이 미분과 연쇄법칙이다. \
이 포스트에서는 평균변화율, 미분계수 및 도함수의 개념과 미분의 종류, 그리고 연쇄법칙에 이르기까지 정리해보았다.

<span style="color:gray">*오류가 있다면 댓글로 무엇이든 지적해주시면 정말 감사드리겠습니다!*</span>


&nbsp;
# 용어 정리 

미분과 이어 그래디언트까지 공부하게 되면, "변화"라는 말을 많이 접하게 된다. 흔히 접하게 될 다양한 변화와 관련된 용어들을 먼저 아래와 같이 정리해보았다:
- <span style="color:orange">**변화율**</span>: **변화의 빠르기** (단위당 변화) &#8594; 변화의 <span style="color:orange">방향과 빠르기</span> 모두 포함
- <span style="color:orange">**변화율의 크기**</span>: 얼마나 **강하게 혹은 가파르게** 변화하는 지 &#8594; 방향을 제외한 순수한 변화의 <span style="color:orange">빠르기</span>만 의미
- <span style="color:orange">**변화가 얼마나 빠른가**</span>: **변화율** 또는 **변화율의 크기**
- <span style="color:orange">**변화량**</span>: 실제로 **얼만큼** 변화했는 지 (총 변화) 
- <span style="color:orange">**변화가 얼마나 큰가**</span>: 문맥 없이 이 표현을 쓰면 수학적으로는 불명확

&nbsp;
# 도함수와 미분
<span style="color:orange">평균변화율</span>이란 한마디로 함수의 기울기이다! \
$y=f(x)$일 때 함수 내 주어진 구간에서 $y$가 $x$에 비해 평균적으로 얼마나 증가하는지, 혹은 한 단위당 얼만큼 증가하는지를 뜻한다. \
기하학적으로는, 함수 내 주어진 구간의 양 끝점을 잇는 직선의 기울기이다:

$$
    \frac{\triangle{y}}{\triangle{x}}=\frac{f(b)-f(a)}{b-a}=\frac{f(a+h)-f(a)}{h}
$$

<span style="color:orange">미분계수(differential coefficient; derivative)</span>란 순간적인 변화율을 의미한다. \
함수 $y=f(x)$에가 있을 때, $x=a$일 때 $x$의 증가량이 0에 가까워질 때의 평균변화율을 의미한다. \
기하학적으로는, 함수 $y=f(x)$ 상에 있는 점 $(a, f(a))$에서의 접선의 기울기를 의미한다. 선과 접한 부분은 아주 작은 부분이므로 결국 **순간변화율**을 의미한다:

$$
    f'(a)=\lim_{\triangle{x}\to0}\frac{\triangle{y}}{\triangle{x}}=\lim_{\triangle{x}\to0}\frac{f(a+\triangle{x})-f(a)}{\triangle{x}}=\lim_{h\to0}\frac{f(a+h)-f(a)}{h}
$$

<span style="color:orange">도함수(derivative)</span>란 함수의 독립변수에 대한 종속변수의 ``변화율``이다. \
함수 $y=f(x)$가 정의한 구간 중에서 미분 가능한 모든 $x$가 정의역일 때, 해당 점에서의 미분계수가 함수값인 **함수**라고 할 수 있다. \
즉, 각 $x$에서의 순간변화율(기울기)을 구한 값을 모은 함수이다:

$$
    f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}=\frac{dy}{dx}=\frac{d}{dx}f(x)
$$


이 때 $dx$는 변수 $x$ 에서 일어나는 아주 작은 변화의 상태, $x$의 변화량이 0을 향해 무한히 다가가고 있는 상태를 의미한다. 이를 **미소변화량**(Infinitesimal Change)이라고도 부르며, 도함수는 결국 <u>$x$의 미소변화량과 $y$의 미소변화량의 비율에 대한 함수</u>라고 볼 수 있다. \
$\frac{dy}{dx}$와 같은 표기법을 라이프니츠식 표기법이라고도 부른다.

<span style="color:orange">미분(differential)</span>이란, 변수 $x$의 변화에 의해 함수값 $f$에서 일어났다고 "추정"되는 변화량을 의미한다. 이는 실제 변화량($\triangle$)과는 다르다는 점을 주목해야 한다. \
변수 $x$에서 미소변화량인 $dx$만큼 변화가 일어났을 때, $x$에서의 순간적인 기울기인 도함수를 기준으로 근사한 함수값 $f$에서의 미소변화량을 뜻하며, 아래와 같이 표현된다:

$$
dy=f'(x)dx
$$

여기에서 또 주의할 점은, 이 "미분"과 우리가 "미분한다"라고 할 때의 미분(differentiation)은 다른 개념이라는 것이다. <span style="color:blue">미분(differentiation)은 도함수를 구하는 연산과정 자체를 뜻한다!</span>

위와 같은 미분의 기초 개념들은 다음 포스트에 배울 "그래디언트"라는 개념을 이해하는 데 필수적이고, 나아가서는 경사하강법을 통해 딥러닝 모델의 오류(cost)을 변화시키는 과정을 이해하는 데에서도 중요하다. 


&nbsp;
# 편도함수와 편미분
<span style="color:orange">편도함수(partial derivative)</span>이란 함수에서 변수 하나에 대해서만 조사한 ``변화율``을 의미한다. \
변수 $x, y$로 이루어진 다변량함수 $f(x,y)$에서, **하나의 변수 $x$ 혹은 $y$만을** 움직였을 때의 함수값의 변화율이다. 관심있는 변수만 하나의 변수로 두고 나머지 변수는 상수로 취급하여 미분하는 방식이다.

미분연산자는 $d$ 표시 대신 $\partial$ 표시로 나타내며, 다변량함수 $f(x,y)$에서의 $x$에 대한 편도함수는 아래와 같이 표현된다:  

$$
f'_x(x,y) = \frac{\partial{f}}{\partial{x}} = \frac{\partial}{\partial{x}}f(x, y)
$$

위 함수의 변수 $x, y$에 들어갈 값은, 축이 2개인 2차원 벡터에 대응될 수 있다. 각 축은 기저 벡터로 표현되어 고유의 방향을 가지고 있기 때문에, 이 벡터는 두 가지의 방향을 가지고 있을 것이다.

[이전 포스트](https://rsy1026.github.io/deep_learning/gradient_descent_vector)에서 설명했듯이, 벡터는 서로 직교하는(독립적인) 기저 벡터들의 선형 결합으로 이루어져 있다. 그렇기 때문에 벡터의 각 기저 축에 대해 독립적으로 변화량을 측정하는 것이 가능하다. 이에 따라 2차원, 3차원과 같은 다차원에서의 복잡한 변화를 각 기저 축의 방향에 따른 단순한 변화의 합으로 표현할 수 있다.

변수 $x$ 혹은 $y$만을 $h$만큼 움직였을 때 각각에 대한 식은 다음과 같다:

$$
\begin{aligned}
    f'_x(x,y)=f'_x(x_0,y_0)=\frac{\partial{f}}{\partial{x}}(x_0,y_0)=\lim_{h\to0}\frac{f(x_0+h,y_0)-f(x_0,y_0)}{h}
\end{aligned}
$$

$$
\begin{aligned}
    f'_y(x,y)=f'_x(x_0,y_0)=\frac{\partial{f}}{\partial{y}}(x_0,y_0)=\lim_{h\to0}\frac{f(x_0,y_0+h)-f(x_0,y_0)}{h}
\end{aligned}
$$

위 식에서 $f_x(x,y)$에 대해 $a_0=[x_0,y_0]$을 사용하여 다시 정리하면 다음과 같다:

$$
    f'_x(a_0)=\lim_{h\to0}\frac{f(a_0+h\hat{i})-f(a_0)}{h}
$$

이 때, $\begin{aligned}\hat{i}=\begin{bmatrix}1\\\\0\end{bmatrix}\end{aligned}$는 $x$축에 대한 단위 벡터이다.


<span style="color:orange">편미분(partial differential)</span>은 앞선 도함수-미분 관계에서와 마찬가지로, 함수값의 특정 방향으로의 실제 ``변화량``을 의미한다. \
즉, 다른 변수를 고정한 상태에서 특정 변수 $x$만 미소변화량 $dx$만큼 변화했을 때, 함수값 $f$가 $x$축의 방향으로 얼만큼 많이 변화했는 지를 뜻한다.

$$
d_xf=\frac{\partial{f}}{\partial{x}}dx
$$

이 때 $dx$가 $\frac{\partial{f}}{\partial{x}}$ 뒤에 곱해지는 이유는, 함수값의 실제 변화량을 구하는 것이기 때문이다. \
$\frac{\partial{f}}{\partial{x}}$는 변수 $x$에 대한 함수값 $f$의 변화율이고, $dx$는 변수 $x$에서 일어나는 미소변화량이라고 했으므로, 두 표현을 곱한다는 것은 실제로 $x$가 움직인 양에 의해 함수값 $f$가 변화한 양을 보는 것이다. \

그리고 "편미분한다"의 편미분(partial differentiation)은 마찬가지로 이 편미분과는 다르며, <span style="color:blue">편미분(partial differentiation)은 편도함수를 구하는 연산 과정 자체를 뜻한다!</span>



&nbsp;
# 전도함수와 전미분
전도함수와 전미분의 관계는 앞서 설명한 도함수-미분이나 편도함수-편미분의 관계와 다르다.


먼저 <span style="color:orange">전미분(total differential)</span>이란, 다변량함수에서 **모든** 변수들이 움직였을 때에 대해 추정된 함수값의 ``변화량``을 의미한다. \
전미분도 미분 및 편미분과 같이 함수 앞에 $d$ 표시를 붙여서 나타내며, 결국 모든 변수의 미소변화량에 대한 표현이라고도 할 수 있다. 

다변량함수 $f(x,y,z,...)$가 있을 때 해당 전미분은 아래와 같이 표현된다:

$$
    df(x,y,z,...)=\Big(\frac{\partial{f}}{\partial{x}}\Big)dx+\Big(\frac{\partial{f}}{\partial{y}}\Big)dy+\Big(\frac{\partial{f}}{\partial{z}}\Big)dz+...
$$

우리가 "전미분한다"라고 할 때 전미분은, 위와 같은 **전미분 식**을 구하는 연산 과정을 뜻한다.

반면 전도함수(total derivative)는 함수의 독립변수에 공통된 매개변수가 있을 때, 그 매개변수에 대한 함수값의 최종적인 ``변화율``을 의미한다. \
다변량함수 $f(x,y)$의 독립변수 $x, y$가 아래와 같이 각각 매개변수 $t$의 함수값으로 정의된다고 가정하자:

$$
    x=f(t), \quad y=g(t), \quad z=f(x, y)
$$

이 때 매개변수 $t$에 대한 전도함수는 아래와 같이 표현될 수 있다:

$$
    \frac{dz}{dt}=\frac{\partial{z}}{\partial{x}}\frac{dx}{dt}+\frac{\partial{z}}{\partial{y}}\frac{dy}{dt}
$$

위 식에서 $\frac{dz}{dt}$에는 $\partial$ 표시 대신 $d$ 표시가 붙었음을 주목하자. $d$가 쓰였다는 것은 함수의 변화율을 결정지을 독립변수가 오직 하나만 있다는 뜻이다. \
반면 편미분에서 쓰이는 $\partial$ 표시는 함수의 변화율을 결정지을 독립변수가 여러 개 존재한다는 뜻으로, 지금 보고자 하는 변수 외 나머지 변수들을 상수 취급하고 있다는 일종의 부분적인 관찰임을 알려주는 역할을 한다.

전미분은 그래디언트와, 전도함수는 연쇄법칙과 연결되기 때문에 사전에 이해가 꼭 필요한 개념이라고 할 수 있다.



&nbsp;
# 연쇄법칙
<span style="color:orange">연쇄법칙(chain rule)</span>이란 위에서 설명한 전도함수를 구하는 방법으로, 합성함수의 미분법이다. \
일변수 합성함수 $y=f(g(x))$에 대해 $t=g(x), y=f(t)$로 분해되어 있다면 다음 식이 성립한다:

$$
    \frac{dy}{dx}=\frac{dy}{dt}\frac{dt}{dx}
$$

즉, 합성함수의 변화율을 겉함수 $f$와 속함수 $g$를 각각 미분하여 곱해주는 방식아며, 이러한 표현 법칙을 연쇄법칙이라고 부른다. \
이 때 합성함수를 이루는 모든 함수들은 모두 미분 가능해야 한다.

이변수 합성함수 $z=f(x,y)$에 대해 $x=g(t), y=h(t)$이고, $f(x,y), g(t), h(t)$ 모두 미분 가능한 함수라면, 아래와 같이 전도함수와 같은 형태의 식이 성립한다: 

$$
    \frac{dz}{dt}=\frac{dz}{dx}\frac{dx}{dt}+\frac{dz}{dy}\frac{dy}{dt}
$$

구체적인 증명법은 이 [링크1](https://blog.naver.com/mindo1103/90103548178) 및 이 [링크2](https://vegatrash.tistory.com/17)에서 확인할 수 있다!

이 연쇄법칙에 의해, 경사하강법을 사용하여 딥러닝 모델 내 여러 개의 레이어에 걸친 파라미터들을 효율적으로 업데이트하는 것이 가능해진다. \
그 업데이트 과정 자체를 역전파(backpropagation)라고 하며, 이와 관련된 내용은 별도의 챕터에 정리하고자 한다.   


&nbsp;
# 정리
- 도함수는 함수의 독립변수에 대한 종속변수의 변화율이고, 미분은 도함수로 추정된 함수값의 변화량이다.
- 편도함수는 함수의 독립변수 하나만 변하고 나머지 변수는 상수일 때에 대한 종속변수의 변화율이고, 편미분은 편도함수로 추정된 함수값의 특정 한 방향으로의 변화량이다. 벡터의 편미분 연산은 벡터가 서로 독립적인 기저벡터의 선형 결합이기 때문에 가능하다.
- 전미분은 함수의 모든 독립변수에 대한 종속변수의 추정 변화량이며, 각 변수의 편미분의 합으로 표현된다. 즉 다차원 벡터의 복잡한 변화를 각 축의 단순한 변화의 합으로 표현할 수 있다.
- 전도함수는 독립변수들의 공통 매개변수에 대한 함수값의 변화율이며, 연쇄법칙은 합성함수의 미분법으로 전도함수를 구하는 데 사용되는 표현 법칙이다. 이 법칙으로 인해 경사하강법과 역전파를 통한 모델 학습을 가능해진다.

&nbsp;
### 참고 자료
- [편미분∂ 전미분d 변화량Δ 그래디언트∇](https://velog.io/@seokjin1013/%ED%8E%B8%EB%AF%B8%EB%B6%84%EA%B3%BC-%EC%A0%84%EB%AF%B8%EB%B6%84)
- [편미분과 전미분](https://velog.io/@swan9405/%ED%8E%B8%EB%AF%B8%EB%B6%84%EA%B3%BC-%EC%A0%84%EB%AF%B8%EB%B6%84)
- [Khan Academy - 평균변화율 복습](https://ko.khanacademy.org/math/algebra/x2f8bb11595b61c86:functions/x2f8bb11595b61c86:average-rate-of-change-word-problems/a/average-rate-of-change-review)
- [기본개념 - 도함수의 정의](https://bhsmath.tistory.com/172)
- [ML 기초 - 수포자가 이해한 미분과 편미분 (feat. 경사하강법)](https://airsbigdata.tistory.com/191)
- [편미분과 전미분](https://velog.io/@swan9405/%ED%8E%B8%EB%AF%B8%EB%B6%84%EA%B3%BC-%EC%A0%84%EB%AF%B8%EB%B6%84)
- [다변수함수의 연쇄법칙(Chain Rule)](https://blog.naver.com/mindo1103/90103548178)
- [8. 연쇄 법칙과 증명 (Chain Rule)](https://vegatrash.tistory.com/17)


&nbsp;
&nbsp;
<script src="https://utteranc.es/client.js"
        repo="rsy1026/rsy1026.github.io-comments"
        issue-term="pathname"
        theme="light-dark"
        crossorigin="anonymous"
        async>
</script>


<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-0DVY3PH8P2"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-0DVY3PH8P2');
</script>