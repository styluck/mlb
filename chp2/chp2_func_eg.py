# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:49:55 2024

@author: 6
"""

#%% 利用生成器得到一个数列的平方


#%% 利用生成器来构建无穷长度的数列


#%% *args 
def my_function(*args):
    for arg in args:
        print(arg)

def sum_numbers(*args):
    return sum(args)

def make_pizza(size, *toppings, crust="regular"):
    print(f"\nMaking a {size}-inch pizza with {crust} crust.")
    if toppings:
        print("Toppings:")
        for topping in toppings:
            print(f"- {topping}")


#%% 和 **kwargs
def my_function(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")

def my_function(**kwargs):
    if "param1" in kwargs:
        print(f"we got param1: {kwargs['param1']}")
    if "param2" in kwargs:
        print(f"we got param2: {kwargs['param2']}")


# 两个混在一起用
def my_function(*args, **kwargs):
    print("Positional arguments:", args)
    print("Keyword arguments:", kwargs)


#%% 解包
def my_function(name, age, city):
    print(f"{name} is {age} years old and lives in {city}.")




#%% 上下文管理器
class MyContextManager:
    def __enter__(self):
        print("Entering the context...")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context...")
        if exc_type:
            print(f"An exception occurred: {exc_value}")
        return True  # Suppresses exceptions if True



#%% 发生错误时，仍然能够自行退出
class MyContextManager:
    def __enter__(self):
        print("Acquiring resource")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print(f"Exception caught: {exc_value}")
        print("Releasing resource")
        return True  # Suppress exception



#%% 如何使用一个装饰器
def my_decorator(func):
    def wrapper(*args, **kwargs):
        # Code to execute before calling the original function
        print("Something before the function")

        # Call the original function
        result = func(*args, **kwargs)

        # Code to execute after calling the original function
        print("Something after the function")

        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# Equivalent to: say_hello = my_decorator(say_hello)
say_hello()

#%% 利用装饰器将一个函数重复n次
def repeat(n):  # n is the argument passed to the decorator
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator


# 多个装饰器
def decorator1(func):
    def wrapper(*args, **kwargs):
        print("Decorator 1")
        return func(*args, **kwargs)
    return wrapper

def decorator2(func):
    def wrapper(*args, **kwargs):
        print("Decorator 2")
        return func(*args, **kwargs)
    return wrapper



