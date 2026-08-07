# Robert Yang 2019 Task Dataset

The task definitions, input and output representations, and population-coding methods described in this document are based on the multitask cognitive-task framework introduced by Yang et al. (2019).

The corresponding task-generation implementation included in this repository is derived from the original [`gyyang/multitask`](https://github.com/gyyang/multitask) codebase.

## fdgo (Full-Duration Go) Task

### Experiment Explanation

A subject faces a two-dimensional display in which possible stimulus locations are arranged as angular positions around a central fixation point.

At the start of each trial:

1. The subject must fixate on a central point.
2. A stimulus appears at a random angular location on a circular ring.
3. The stimulus remains visible during the stimulus and response periods.
4. When the fixation signal turns off, the subject must respond toward the stimulus location.

The `fdgo` task evaluates the ability to maintain fixation until the go cue and then produce a directional response toward a continuously available stimulus.

---

## Data

### Input Data Structure

The input feature dimension is 86.

The `fdgo` task uses the following input channels:

1. **Fixation input: 1 channel**

   A scalar value indicating whether the fixation signal is active.

2. **Stimulus ring inputs: 32 channels x 2**

   Two population-coded sensory rings representing stimulus directions.

3. **Rule input: 21 channels**

   A one-hot vector indicating which task rule is active. Exactly one of the 21 rule channels is active, while the remaining channels are zero.

The total input dimension is:

```text
1 + 32 + 32 + 21 = 86
```

---

### Output Data Structure

The output dimension is 33.

The output consists of:

1. **Fixation output: 1 channel**

   Indicates whether fixation should be maintained.

2. **Target response direction: 32 channels**

   A population-coded representation of the direction toward which the model should respond.

The total output dimension is:

```text
1 + 32 = 33
```

---

## Stimulus Encoding Method: Population Coding

The stimulus direction is represented by the angular variable `theta`.

The model uses 32 sensory units with preferred directions distributed uniformly around a circle. Unit `i` has the preferred direction:

```text
preferred_direction_i = 2 * pi * i / 32
```

where:

```text
i = 0, 1, ..., 31
```

To encode a stimulus direction `theta`, the circular distance between `theta` and the preferred direction of unit `i` is computed as:

```text
distance_i = min(
    abs(theta - preferred_direction_i),
    2 * pi - abs(theta - preferred_direction_i)
)
```

The activation of unit `i` is then determined using a Gaussian-like tuning function:

```text
x_i(theta) = A * exp(
    -(distance_i ** 2) / (2 * sigma ** 2)
)
```

where:

* `theta` is the stimulus direction;
* `preferred_direction_i` is the preferred direction of unit `i`;
* `A` is the peak response amplitude;
* `sigma` controls the tuning width.

Common parameter values are:

```text
A = 0.8
sigma = pi / 8
```

This encoding produces a smooth bump-shaped activity pattern across the 32 sensory units, centered on the stimulus direction.

---

## Decoding the Stimulus Direction

Given the activities `x_i` of the 32 directional units, the represented direction can be decoded using the population-vector method.

First, compute the weighted Cartesian components:

```text
s_x = sum(
    x_i * cos(preferred_direction_i)
)
```

```text
s_y = sum(
    x_i * sin(preferred_direction_i)
)
```

The decoded direction is then:

```text
theta_hat = atan2(s_y, s_x)
```

This gives the circular mean of the directions represented by the population activity.

---

## Summary

* `fdgo` is a stimulus-guided directional-response task.
* The model must maintain fixation until the go cue.
* The stimulus remains available until the response period.
* Stimulus direction is represented using 32 directionally tuned units.
* Encoding produces a smooth population-activity bump around the stimulus direction.
* Decoding uses a population vector and the `atan2` function.
* The task framework is based on Yang et al. (2019).

---

## Reference

Yang, Guangyu Robert, et al. “Task Representations in Neural Networks Trained to Perform Many Cognitive Tasks.” *Nature Neuroscience*, vol. 22, no. 2, 2019, pp. 297–306.
