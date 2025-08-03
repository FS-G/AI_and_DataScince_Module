# Python Recursion & Lambda Functions - Simple Guide

## What is Recursion? 🔄

**Recursion** is when a function calls itself! It's like looking into two mirrors facing each other - you see the same thing over and over again.

### Why use recursion?
Think of it like climbing stairs. To get to the 5th step, you need to:
1. Get to the 4th step first
2. Then take one more step up

### Simple Recursion Example: Counting Down

```python
def countdown(n):
    if n == 0:  # This is the "stop" condition
        print("Blast off! 🚀")
    else:
        print(n)
        countdown(n - 1)  # Function calls itself!

# Try it!
countdown(5)
```

**Output:**
```
5
4
3
2
1
Blast off! 🚀
```

### Another Example: Adding Numbers

```python
def add_numbers(n):
    if n == 1:  # Stop when we reach 1
        return 1
    else:
        return n + add_numbers(n - 1)  # Add current number + sum of smaller numbers

# Calculate 1+2+3+4+5
result = add_numbers(5)
print(f"1+2+3+4+5 = {result}")  # Output: 15
```

---

## What are Lambda Functions? ⚡

**Lambda functions** are tiny, one-line functions. They're like shortcuts for simple tasks!

### Regular Function vs Lambda Function

**Regular way:**
```python
def double(x):
    return x * 2

print(double(5))  # Output: 10
```

**Lambda way:**
```python
double = lambda x: x * 2
print(double(5))  # Output: 10
```

### More Lambda Examples

**Adding two numbers:**
```python
add = lambda a, b: a + b
print(add(3, 7))  # Output: 10
```

**Check if number is even:**
```python
is_even = lambda x: x % 2 == 0
print(is_even(4))  # Output: True
print(is_even(5))  # Output: False
```

**Find the bigger number:**
```python
bigger = lambda a, b: a if a > b else b
print(bigger(10, 5))  # Output: 10
```

### Using Lambda with Lists

**Double all numbers in a list:**
```python
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
print(doubled)  # Output: [2, 4, 6, 8, 10]
```

**Filter even numbers:**
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # Output: [2, 4, 6, 8, 10]
```

---

## Fun Practice Activities! 🎯

### Recursion Practice:
Try making a function that prints your name n times!

```python
def print_name(name, times):
    if times == 0:
        print("Done!")
    else:
        print(name)
        print_name(name, times - 1)

print_name("Alice", 3)
```

### Lambda Practice:
Create a lambda function that converts Celsius to Fahrenheit!

```python
celsius_to_fahrenheit = lambda c: (c * 9/5) + 32
print(celsius_to_fahrenheit(25))  # Output: 77.0
```

---

## Key Points to Remember 📝

**Recursion:**
- A function that calls itself
- Always needs a "stop" condition
- Good for breaking big problems into smaller ones

**Lambda Functions:**
- Short, one-line functions
- Great for simple operations
- Often used with `map()`, `filter()`, and `sort()`

## Quick Tips ✨
- **Recursion:** Think "What's the smallest version of this problem?"
- **Lambda:** Think "Can I write this function in one line?"

Happy coding! 🐍✨