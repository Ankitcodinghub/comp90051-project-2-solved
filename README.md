# comp90051-project-2-solved
**TO GET THIS SOLUTION VISIT:** [COMP90051 Project 2 Solved](https://www.ankitcodinghub.com/product/comp90051-statistical-machine-learning-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;120109&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;COMP90051 Project 2 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Project 2 Description

Most machine learning models are based on supervised learning, applied to a given training dataset which includes instances and their labels. In many practical settings, however, labelled data is not readily available. While unsupervised learning is one way of dealing with this situation, a popular alternative that can still make use of powerful discriminative machine learning methods is Active Learning. This describes a machine learning framework whereby data is labelled in an incremental manner, and a machine learning model is used to decide what data instances should be annotated next. This allows the model to focus the labelling effort—which can often be slow and expensive, e.g., requiring manual human labour—on highly informative instances which can correct weaknesses in the model. There are several different strategies for choosing (or synthesising) data for annotation, and this project will require you to understand and implement several such techniques.

In this project, you will work individually (not in teams) to implement several active learning methods. For all of these methods you will have to refer to the literature, and read and understand it yourself. By the end of the project you should have developed

ILO1. An understanding Active Learning, and how it relates to other learning paradigms in machine learning;

ILO2. Better understanding of how machine learning concepts like uncertainty can support machine learning;

ILO3. Demonstrable ability to implement ML approaches in code; and

ILO4. An ability to pick up recent machine learning publications in the literature, understand their focus, contributions, and algorithms enough to be able to implement and apply them. (And being able to ignore other presented details not needed for your task.)

Overview

We will be working largely from the following book which presents a detailed survey of the field of active learning:

in particular chapters 2, 3 and 5. While the above book does a superb job of covering many of the algorithms and theory, you will also want to read other papers, such as the original papers presenting the algorithms, as well as papers with a stronger empirical focus like (Schein and Ungar, 2007) as cited in the above book.

Your task is to develop pool-based active learning methods to implement an active learning framework, and using this to support uncertainty sampling, query by committee and hierarchical cluster-based sampling. In all cases logistic regression is to be used as the component classification model.

You will be using the dataset to simulate active learning of a classifier which detects the language given a handwritten character image. We will use part of the OmniGlot dataset which comprises the alphabets of 30 different languages, with a total of 19,280 images of hand-drawn characters. This will be used for simulating active learning by starting from a small seed set of 300 labelled images,3 then progressively revealing more labels over the course of an active learning run, retraining a classifier at each stage. The testing set will be used for reporting of results. Unlike the previous project, we have provided you with all the data, so you will need to be careful you use these resources in an appropriate manner.

Required Resources The LMS page for project 2 comprises

• project2.pdf this spec;

• project2.ipynb Jupyter notebook skeleton in Python;

• project2 dataset.ipynb Jupyter notebook illustrating the dataset and its processing (purely FYI);

• images.npy numpy file of vector embeddings of the dataset instances; and

• labels.npy numpy file with a vector of labels, one per instance.

You will implement code in Python Jupyter notebooks, which after running on your machine you will submit via LMS. Further detailed rules about what is expected with code are available towards the end of this spec. We appreciate that while some have entered COMP90051 with little prior Python experience, many workshops so far have exercised and built up basic Python and Jupyter knowledge.

Dataset: The LMS page for project 2 contains a dataset suitable for validating your AL algorithms. You should download these files and familiarise yourself with their content:

• images.npy: a matrix of 19,280 x 1,000 encoding images of handwritten characters, each embedded in 1000d

• labels.npy: a vector of 19,280 string valued labels, encoding to the alphabet name of the corresponding image

The notebook includes code for processing these files into a training pool, seed set, and testing set. Note that the seed set lists the indices of examples from the training pool that are to be treated as labelled, while all other instances are to be initially considered unlabelled. Over the course of an active learning simulation additional instances will have their labels revealed, and become part of the labelled set.

Implement Python functions train logistic regression and evaluate logistic regression accuracy which train a logistic regression classifier, and evaluate it against a test set. You are welcome to use python libraries, e.g., those you’ve encountered in the workshops, for this task. Use these two functions to train logistic regression over the seed set, and train on the full pool. Report the accuracy on the test set. This will serve as a rough guide for the range of performance you might expect to see the active learning models, below.

Your task is to implement the pool based active learning algorithm, following the pool based active learning function supplied in the notebook, and the random selection heuristic in the method random select (such that

1: U = pool of unlabelled instances, {x}

2: L = set of initial labelled instances, {hx,yi}

3: b = number of instances to label in each step

4: for t = 1,2,…,T do

5: θ(t) = train(L)

6: score all instances in pool, r = select(U)

7: for all j ∈ argmax(b,r) do

8: reveal label yj

9: add hxj,yji to L

10: remove xj from U

11: end for

12: end for

13: return

Figure 1: Generic active learning algorithm, adapted from Figure 2.3 of Settles (2012). Requires a train function and select function, and argmax(b,r) returns the indices of the b maximising elements of r (tied values to be broken at random).

lines 6-7 end up sampling uniformly without replacement from the pool). Apply this active learning function along with the random selection function and the training function from Part 1. You should use run active learning such that it starts with the supplied seed set, and then trains on progressively larger datasets growing by b = 60 instances in each step, and stopping after reaching 3000 instances, i.e., T = 45 steps. Present the results of your experiment in a plot showing test accuracy compared to number of training instances.

Now that you have a working infrastructure, it is time to develop the first active learning algorithm. For this, you will use uncertainty sampling, one of the simplest and most enduring methods for active learning. This works by selecting instances that have the most predictive uncertainty under the model, and accordingly one would expect the model to learn the most from their labelling.

A.I. Schein and L.H.Ungar. ‘Active learning for logistic regression: An evaluation’. Machine Learning, 68(3):235265, 2007. https://repository.upenn.edu/cgi/viewcontent.cgi?article=1378&amp;context=cis_papers

Your task is to implement the entropy selection heuristic in the method logistic regression entropy select. Evaluate your selection function in the active learning framework and present a plot of the accuracy over the active learning run, based on the same run parameters from Part 2.

The next active learning algorithm is based on training an ensemble of several models and measuring the extent of disagreement within the ensemble as the selection heuristic. The idea is that instances that are predicted differently by each ensemble member are likely to be highly informative, once labelled. You will need to review Settles chapter 3, specifically section 3.4 on ‘Query by Committee’, and the several cited papers on pages 28-29. Your task is to implement the QbC method. You should set the size of the ensemble to two, and use bagging as the ensembling method.

You should implement three techniques for measuring disagreement: vote entropy, soft vote entropy and the KL-based method (Settles eq 3.1-3.3, page 29), implementing each of these as a separate selection function in methods query by committee xxx where xxx is replaced with the name of each of the three methods. You will also need to complete the specialised training function, train committee to learn an bagged ensemble of classifiers in each step.

The last algorithm required in the project, is a hierarchical sampling method presented in the following paper:

S.Dasgupta and D.J.Hsu. ‘Hierarchical sampling for active learning’. In Proceedings of the International Conference on Machine Learning (ICML), pages 208215. ACM, 2008. https://icml.cc/Conferences/2008/papers/324.pdf

Project Submission
