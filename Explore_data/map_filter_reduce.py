from time import time_ns

num_repeat = 100000

def cubed(x):
        return x**3

lambda_cubed = lambda x:x**3

start = time_ns()
for i in range(num_repeat):
        cubed(9)

print(f"Cube of 9 is {cubed(9)}")
print(f"Time taken for previous method : {(time_ns()-start)//1_000/num_repeat} ms")

start = time_ns()
for i in range(num_repeat):
        lambda_cubed(9)
print(f"Cube of 9 is {lambda_cubed(9)}")
print(f"Time taken for previous lambda : {(time_ns()-start)//1_000/num_repeat} ms")

start = time_ns()
for i in range(num_repeat):
        (lambda x:x**3)(9)
print(f"Cube of 9 is {(lambda x:x**3)(9)}")
print(f"Time taken for inline lambda : {(time_ns()-start)//1_000/num_repeat} ms")

########################################################################################################################
# Map
########################################################################################################################
start = time_ns()
for i in range(num_repeat):
        squares=[]
        for i in range(1,11):
                squares.append(i**2)
print(squares)
print(f"Time taken for a for loop to generate squares: {(time_ns()-start)//1_000/num_repeat} ms")


start = time_ns()
for i in range(num_repeat):
        squares=list(map(lambda i:i**2,list(range(1,11))))

print(squares)
print(f"Time taken for map and lambda function to generate squares : {(time_ns()-start)//1_000/num_repeat} ms")

########################################################################################################################
# Filter
########################################################################################################################
start = time_ns()
for i in range(num_repeat):
        evens=[]
        for i in range(1,11):
                if i%2==0:
                        evens.append(i)
print(evens)
print(f"Time taken for a for loop : {(time_ns()-start)//1_000/num_repeat} ms")


start = time_ns()
for i in range(num_repeat):
        squares=filter(lambda i:i%2==0,list(range(1,11)))
print(evens)
print(f"Time taken for filter : {(time_ns()-start)//1_000/num_repeat} ms")

########################################################################################################################
# Reduce
########################################################################################################################
