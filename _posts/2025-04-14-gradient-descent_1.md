---
title: "[처음부터 이해하기] 경사하강법 (Gradient Descent) (1) - 벡터"
categories: 
    - deep_learning
tags: 
    - machine learning
    - deep learning
    - vector
    - vector space

permalink: deep_learning/gradient_descent_vector

toc: true
toc_sticky: true
layout: single
sitemap:
  changefreq: daily
  priority : 1.0

date: 2025-04-14
last_modified_at: 2026-03-18
---

경사하강법을 처음부터 이해하기 위한 가장 중요한 개념 중 하나가 바로 "벡터"가 아닌가 싶다. \
이 포스트에서는 벡터와 벡터 공간의 정의, 직교좌표계, 그리고 경사하강법에서 중요한 벡터의 특성까지 정리해보았다.

<span style="color:gray">*오류가 있다면 댓글로 무엇이든 지적해주시면 정말 감사드리겠습니다!*</span>

<!-- &nbsp;
# 용어 정리 

그래디언트라는 개념을 처음 공부할 때 "변화"라는 말을 많이 보게 됩니다. 이 때 '변화율', '변화량', '변화가 얼마나 빠른가' 등 변화와 관련된 용어들이 종종 혼재되어 쓰이곤 해서 개인적으로 엄청 헷갈렸더랬습니다. 그래서 그래디언트를 설명할 때 자주 쓰이는 변화 관련 용어들을 정리해보았습니다.
- <span style="color:orange">**변화율**</span>: **변화의 빠르기** (단위당 변화) &#8594; 변화의 <span style="color:orange">방향과 빠르기</span> 모두 포함
- <span style="color:orange">**변화율의 크기**</span>: 얼마나 **강하게 혹은 가파르게** 변화하는 지 &#8594; 방향을 제외한 순수한 변화의 <span style="color:orange">빠르기</span>만 의미
- <span style="color:orange">**변화가 얼마나 빠른가**</span>: **변화율** 또는 **변화율의 크기**
- <span style="color:orange">**변화량**</span>: 실제로 **얼만큼** 변화했는 지 (총 변화) 
- <span style="color:orange">**변화가 얼마나 큰가**</span>: 문맥 없이 이 표현을 쓰면 수학적으로는 불명확

여기서 가장 헷갈리는게 **변화율**과 **변화량**일 것 같습니다. 변화가 얼마나 빠른지(변화율)와 얼마나 많이(변화량) 발생했는지가 다른 개념이라는 부분을 명심하면 좋을 것 같습니다. 그런 점에서 '변화가 크다'라는 말은 '빠름'과 '많음'을 모두 의미할 수 있기 때문에 문맥적 파악이 필요할 수 있습니다. -->


&nbsp;
# 벡터와 벡터 공간
<span style="color:orange">벡터</span>는 일반적으로 크기와 방향을 갖는 양으로 정의된다.

수학적 정의에 따르면 벡터가 존재하는 공간, 벡터들이 모여있는 집합을 <span style="color:orange">벡터 공간(vector space)</span>이라고 한다. 

벡터 공간이란, <span style="color:orange">벡터 간 덧셈 또는 벡터와 스칼라(크기만 존재) 간 곱셈 연산</span>이 가능하고, 이 연산들에 대해 닫혀 있으며, \
이 연산들과 관련된 8가지 공리(vector space axioms)를 만족하는 모든 벡터들의 집합을 의미한다. 

'닫힌' 연산이란, 연산의 결과가 연산의 입력과 같은 집단에 속한다는 것이다. \
즉, 벡터 공간에 존재하는 두 원소(벡터)로 아까 그 두 가지의 연산을 하면, 마찬가지로 같은 벡터 공간에 존재하는 원소(벡터)가 나오는 것이다.  

벡터 공간은 곧 <span style="color:orange">선형 공간(linear space)</span>이라고도 한다. 벡터들에 스칼라 곱을 해서 더하는 형태를 <span style="color:orange">선형 결합</span>이라고 하며, \
이 선형 결합을 유지한 채 공간을 이동, 회전, 확대, 축소하는 것을 <span style="color:orange">선형 변환</span>이라고 한다. \
선형 변환을 하고 난 결과물도 여전히 벡터 공간에 존재하는 것이다! (*벡터가 곧 선형대수다...!*) 


&nbsp;
# 벡터의 형태
벡터와 벡터 공간 모두 추상적이고, 대수적인 개념이다. 

벡터는 위에서 언급한 규칙들만 만족한다면 **어떤 것이든** 될 수 있고, 숫자 배열 뿐만 아니라, 행렬, 함수, 수열 등 다양한 형태로도 이루어질 수 있다! \
예를 들면 행렬로도 서로 덧셈, 스칼라 곱을 하고 나서도 여전히 행렬이고, 함수도, 수열도 마찬가지이다. 

행렬에서 덧셈, 스칼라 곱을 거쳐서도 행렬이다:

- 요소: $$A, B \in M_{m \times n}$$
- 연산 규칙:
    - 덧셈: $$(A + B)_{ij} = A_{ij} + B_{ij}$$
    - 스칼라 곱: $$(cA)_{ij} = c \cdot A_{ij}$$
- 증명 수식:
    
$$A + B = \begin{bmatrix} a_{11}+b_{11} & a_{12}+b_{12} \\ a_{21}+b_{21} & a_{22}+b_{22} \end{bmatrix} \in M_{2 \times 2}$$

$$cA = \begin{bmatrix} ca_{11} & ca_{12} \\ ca_{21} & ca_{22} \end{bmatrix} \in M_{2 \times 2}$$

수열에 대해서도 마찬가지이다. 무한 수열의 집합 $V = { {a_n} \mid a_n \in \mathbb{R} }$이 있을 때:

- 요소: 수열 $$a = \{a_1, a_2, \dots\}, b = \{b_1, b_2, \dots\}$$, 스칼라 $$\alpha, \beta$$
- 연산 규칙:
    - 덧셈: $$a + b = \{a_1+b_1, a_2+b_2, \dots\}$$
    - 스칼라 곱: $$a = \{\beta a_1, \beta a_2, \dots\}$$
- 증명 수식:
    
$$\alpha a + \beta b = \{ \alpha a_1 + \beta b_1, \alpha a_2 + \beta b_2, \dots, \alpha a_n + \beta b_n, \dots \}$$
    

이 결과도 하나의 무한 수열이 되므로, 벡터 공간의 원소가 될 수 있다. 이는 특히 딥러닝에서 <span style="color:blue">시계열 데이터 (오디오, 텍스트 등)</span>을 다루는 근거가 된다.

함수에서도 구간 $[a, b]$에서 정의된 연속 함수의 집합 $C[a, b]$이 있을 때:

- 요소: 함수 $$f(x), g(x)$$
- 연산 규칙:
    - 덧셈: $$(f + g)(x) = f(x) + g(x)$$
    - 스칼라 곱: $$(cf)(x) = c \cdot f(x)$$
- 증명 수식:
    
    $$h(x) = \alpha f(x) + \beta g(x)$$
    
두 연속 함수의 선형 결합인 $h(x)$ 역시 연속 함수이므로 $h(x) \in C[a, b]$이다.


&nbsp;
# 기저 벡터
벡터는 <span style="color:orange">기저 벡터(basis vector)</span>들의 합으로 나타낼 수 있다.

기저 벡터로 이루어져 있는 집합을 기저(basis)라고 하며, 이들은 크게 세 가지 특징을 가진다:
- <span style="color:orange">선형 독립 (linear independnece)</span>: 기저 내의 어떤 벡터도 다른 벡터들의 선형 결합으로 표현될 수 없어야 한다. 기저 벡터 간 중복된 정보가 전혀 없어야 하고, 이렇게 되면 서로 겹치지 않는 최소한의 벡터들만이 기저를 이룰 수 있다.
- <span style="color:orange">생성 (span)</span>: 기저 벡터들의 선형 결합을 통해 해당 벡터 공간 내 모든 벡터를 생성할 수 있어야 한다. 
- <span style="color:orange">유일성 (uniqueness)</span>: 해당 벡터 공간 내 임의의 벡터는 단 한 가지 방식의 기저 벡터들의 선형 결합으로만 표현될 수 있다. 

모든 벡터 공간에는 어떤 형태로든 기저가 존재한다. 그리고 벡터 공간 안에 수많은 기저(집합)들이 존재할 수 있다!


&nbsp;
# 좌표 공간
기저를 설명할 때는 일반적으로 공간 내 모든 점들을 숫자(좌표)로 나타낼 수 있는 <span style="color:orange">좌표 공간 (coordinate space)</span> $\mathbb{R}^n$이 많이 사용된다. 

그 이유는, 유한한 차원의 벡터 공간은 같은 차원 수의 좌표 공간 $\mathbb{R}^n$와 구조적으로 동일한 **동형(isomorphism)**이기 때문이다. \
(*다만 여기서의 좌표 공간은 거리, 각도 등의 <u>어떠한 기하학적인 약속도 정의되어 있지 않은 실수 좌표 공간</u>*)

유한한 $n$차원의 벡터 공간 $V$에 기저 $B=\{v_1, v_2, ..., v_n\}$이 존재한다면, 모든 벡터 $v \in V$는 기저 벡터들의 선형 결합으로 표현될 수 있다: 

$$
v=c_1v_1+c_2v_2+...+c_nv_n
$$ 

벡터 $v$를 좌표 공간 $F^n$ 내 좌표 순서쌍 ($c_1, c_2, ..., c_n$)으로 매핑할 수 있다면, \
이는 벡터 공간 $V$과 좌표 공간 $F^n$ 간 1대1 대응이 되는 동형 사상(isomorphism)이 존재한다고 간주하는 것이다.

즉, 벡터 공간에 유한한 수의 기저가 존재한다면 그 벡터 공간은 유한한 차원이 되고, 이 때 같은 차원의 좌표 공간과 구조적으로 동일하다.

결과적으로, 벡터가 어떤 형태로 되어 있든 차원이 유한하게 고정되어 있다면 이를 숫자의 나열(좌표)로 변환하여,\
**컴퓨터의 배열이나 텐서(tensor)**에 저장하고 연산할 수 있다는 것을 의미한다. 


&nbsp;
# 유클리드 공간

좌표 공간에서 거리, 각도를 계산할 때 피타고라스의 정리를 많이 사용하는데, \
그게 가능하다는 것은 좌표 공간에 <span style="color:orange">내적(inner product)</span>이라는 연산 규칙이 정의되어 있다는 뜻이다.

즉, 거리와 각도를 정의하기 위해 내적을 사용한다. (내적이 있기 때문에 거리와 각도가 있는 것!)

내적은 한 벡터를 다른 벡터 위로 투영(projection) 혹은 정사영시킨 길이와 그 투영된 벡터의 크기를 곱하는 연산을 의미한다. \
기하학적으로는 두 벡터 간 크기와 방향이 일치하는 지를 보는 것이므로, 두 벡터 사이의 거리 혹은 연관성을 의미한다.

내적이 정의되어 있는 공간을 우리는 <span style="color:orange">내적 공간(inner product space)</span>라고 하고, \
그 상위에는 두 점간의 거리 혹은 벡터의 크기만이 정의되어 있는 <span style="color:orange">노름 공간(normed vector space)</span>이 있다. \
노름 공간에서는 각도가 정의되어 있지 않은 것이다. (노름(norm)이라는 개념 자체가 벡터의 크기를 의미)

내적 공간에서 차원이 유한하고 우리가 아는 표준 내적을 사용한다면, 이를 최종적으로 <span style="color:orange">유클리드 공간(Euclidean space)</span>라고 한다. \
이 공간이 비로소 기하학적으로 모든 개념들이 정의된 공간이다. 

- 벡터 공간 > 노름 공간 > 내적 공간(유클리드 공간)

거리 공식 중 1-norm($L_1$ norm, Manhattan norm)은 노름 공간에 속하고, 2-norm($L_2$ norm, Euclidean norm)은 내적 공간에 속한다.


&nbsp;
# 벡터의 성분과 단위 벡터

만약 원점이 $O$인 2차원 평면좌표가 있을 때, $A(a_1,a_2)$를 종점으로 하는 임의의 벡터 $\overrightarrow{a}$에 대해 다음과 같이 표현할 수 있다:
<!-- ![image info](/assets/images/vector.png){: width="60%"}
*이 [링크](https://jwmath.tistory.com/490)의 이미지를 재구성했습니다.* -->

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

이 때, 실수 $a_1, a_2$는 각각 $x$축과 $y$축의 <span style="color:orange">성분</span>이다. \
$\overrightarrow{e_1},\overrightarrow{e_2}$는 각각 $x$축과 $y$축의 방향을 가지고 두 점 $E_1(0,1)$와 $E_2(1,0)$을 종점으로 하는 크기가 1인 벡터, <span style="color:orange">단위 벡터(unit vector)</span>이다. 

이 단위 벡터들이 곧 기저 벡터이다. 

즉, 벡터 $\overrightarrow{a}$는 각 축에 대한 단위 벡터의 방향과 각각 $a_1, a_2$의 크기를 가지는 두 벡터 $\overrightarrow{OA_1}, \overrightarrow{OA_2}$의 합이다.

$A(a_1,a_2)$를 종점으로 하는 평면 벡터 $\overrightarrow{a}$의 **크기**는 다음과 같이 피타고라스의 정의에 의해 나타낼 수 있다:

$$
    \vert{\overrightarrow{a}}\vert=\sqrt{(a_1)^2+(a_2)^2}
    $$

만약 3차원 좌표계가 있을 때의 임의의 벡터 $\overrightarrow{v}$는 다음과 같이 $x,y,z$축 각각에 대한 성분과 단위 벡터 $\hat{i},\hat{j},\hat{k}$를 이용해 표현할 수 있다:

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

이 때, 3차원 벡터 $\overrightarrow{v}$의 크기는 마찬가지로 [피타고라스 정의](https://kukuta.tistory.com/152)를 사용하여 나타낼 수 있다:

$$
    \vert{\overrightarrow{v}}\vert=\sqrt{a^2+b^2+c^2}
$$


&nbsp;
# 직교좌표계

유클리드 공간 중에서도 모든 축이 서로 직각으로 교차(직교)하는 공간을 <span style="color:orange">직교좌표계(orthogonal coordinate system)</span>라고 힌다. \
그 중 축이 직선인 <span style="color:orange">데카르트 좌표계(Cartesian coordinate system)</span>가 대표적이다. 

벡터를 설명할 때 이 직교좌표계를 널리 사용한다.

기억해야할 점은 기저 벡터들이 <u>서로 선형 독립이라고 해서 무조건 직교하는 것이 아니다</u>! \
다만 기저 벡터들이 직교하지 않는다면, 벡터를 선형 결합으로 표현하는 데 있어 연산이 상당히 골치가 아파진다. 

임의의 벡터 $v$를 두 기저 벡터$e_1, e_2$의 선형 결합으로 표현하려면 아래의 형태와 같다:

$$v=c_1e_1+c_2e_2$$

우리는 주어진 벡터 $v$를 만들 수 있는 성분 $c_1, c_2$를 찾아야 한다. \
이 때 기저 벡터들이 서로 90도이면, 두 벡터를 내적할 때 $\cos 90^\circ=0$으로 인해 각 성분이 속한 항이 0으로 소거될 수 있다. \
예를 들어 성분 $c_2$를 소거하기 위해 기저 벡터 $e_2$와 직교하는 $e_1$를 모든 항에 내적한다:

$$\langle v, e_1 \rangle = c_1 \langle e_1, e_1 \rangle + c_2 \langle e_2, e_1 \rangle$$


그럼 $\langle e_2, e_1 \rangle$ 항은 0이 되면서 $c_2$는 사라지게 되고, $c_1$에 대한 식만 남는다.

기저 벡터 중 일부라도 서로 직교하지 않는다면, 투영을 해도 내적값이 0으로 소거되지 않으므로, \
성분 하나를 구할 때마다 여러 기저 벡터의 간섭을 동시에 고려하는 복잡한 연립방정식을 풀어야 하기 때문에 연산이 매우 어려워진다.

반면에 모든 기저 벡터들이 서로 직교하면 내적 투영을 통해 구하고자 하는 성분만 남겨서 각 성분을 독립적이고 쉽게 찾을 수 있다.

또한 벡터의 성분을 이루는 기저 벡터들은 <u>직교성과 관계없이 서로 선형 독립</u>이기 때문에, \
벡터의 각 성분에 대한 덧셈, 스칼라 곱 등과 같은 선형 연산이 성분별로 독립적으로 이루어질 수 있다. 

이는 이후 경사하강법을 설명할 때 중요하게 다루는 특성이 된다.



&nbsp;
# 벡터와 벡터 공간의 의미

벡터 공간은 일종의 시뮬레이션 공간과 같다. 

실제 세상에는 여러 물리적인 현상들과 오늘날 특히 수많은 복잡한 데이터들이 존재하는데, \
이들을 특정 연산 법칙들을 만족하는 통제된 공간에 배치하면 비교적 쉽게 다룰 수 있게 된다. 

예를 들면 텍스트 데이터를 벡터 공간에 뿌려놓으면, 거리 공식을 통해 단어 간 유사성을 구할 수 있게 되는 것이다. 

따라서 이 벡터와 벡터 공간은 수학, 물리학, 유체 동역학, 신호 처리, 컴퓨터 그래픽스, \
그리고 오늘날 특히 머신러닝과 딥러닝 등 다양한 분야에서 주요하게 다뤄진다. 



&nbsp;
# 정리

- 벡터는 **크기와 방향**을 나타내는 양이며, 벡터가 이루고 있는 벡터 공간은 실제 세상의 복잡한 데이터를 배치해서 다양한 연산을 할 수 있는 **시뮬레이션 공간**이다.
- 벡터는 **기저 벡터들의 선형 결합**으로 나타낼 수 있으며, 유클리드 공간 중 유한한 수의 기저 벡터가 직교하는 **직교좌표계**에서 설명할 때 벡터의 선형 연산이 용이해진다.
- 벡터의 각 성분을 나타내는 단위 벡터(기저 벡터)들이 서로 **선형 독립**이기 때문에, 선형 연산이 각 성분별로 가능해지게 되고, 이는 경사하강법에서 중요한 특성이다.


&nbsp;
### 참고 자료
- [[선형대수] 벡터공간(vector space), 벡터 부분공간(vector subspace), 생성공간(span), 차원(dimension)](https://rfriend.tistory.com/173)
- [선대 10주 . 벡터공간에 대한 개념 이해](https://erase-jeong.tistory.com/75)
- [벡터의 성분](https://jwmath.tistory.com/490)
- [벡터의 성분과 단위 벡터](https://m.blog.naver.com/seolgoons/222031443313)
- [선형 독립(Linearly Independence)과 선형 종속(Linearly Dependence)](https://kxngmxn.tistory.com/24)
- [[선형 대수] 기저 (basis), 랭크 (rank)](https://minair.tistory.com/74)
- [[선형대수] 선형독립(linearly independent), 선형종속(linearly dependent)](https://rfriend.tistory.com/163)
- [[기하학 (Geometry)] 내적 (Inner Product) 란?](https://m.blog.naver.com/sw4r/221939046286)


&nbsp;
&nbsp;
<script src="https://utteranc.es/client.js"
        repo="rsy1026/rsy1026.github.io-comments"
        issue-term="pathname"
        theme="github-light"
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