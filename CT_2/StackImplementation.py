# Stack Implementation function 
# Basically there are 4 functions for implementation of stack
# 1. PUSH , 2. PUSH , 3. PEEK, 4. DISPLAY

# First We need to define the stack 

print ("Stack Implementation")
print("1 For Pushing an element into the stack")
print("2 for Poping an element from the stack")
print("3 for Peeking an element from the stack")
print ("4 displaying the whole stack")
Stack = [] # Which is basically an empty list

def IsEmpty (stk):
    if stk == []:
        return True
    else : 
        return False

def Push(stk, elm):
    stk.append(elm) # Simple Append Function which add an element
    print("Element Inserted ...", stk)

def Pop(stk): # For Pop first we need to check the no.of elements in Stack
    if IsEmpty(stk):
        print("Empty Stack")
    else :
        stk.pop() # Simple Poping out the first Element as it is based on LIFO (Last in First Out)
        print(stk)

def Peek(stk):
    if IsEmpty(stk):
        print("Empty Stack")
    else: 
        print("First element of the stack is", stk[-1])
        print(stk)

def Display(stk):
    if IsEmpty(stk):
        print("Kya be smjh nahi araha hai kya ??")
    else :
        print("Here is your stack",stk[::-1])

while True :
    ch = int(input("Enter your choice : "))
    if ch == 1 :
        print("Push Method")
        PushElem = input("enter Desired Element : ")
        Push(Stack, PushElem)
    elif ch == 2 :
        print ("Pop Method")
        Pop(Stack)
    elif ch == 3 :
        print("Peek Method")
        Peek(Stack)
    elif ch == 4:
        print("Display Method")
        Display(Stack)
    elif ch == 5 :
        break
        
