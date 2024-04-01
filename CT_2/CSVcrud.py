## All the Crud Operation in One File
import pickle
import csv
## 1. Binary File Crud Operation
## Write a Program for crud operation for a shoe store management



# while True:
#     print("1. Add Record ")
#     print("2. Display Record ")
#     print("3. Search Record ")
#     print("4. Exit")

#     ch = int(input("Enter your operation : "))
#     l = []

#     if ch == 1: # Appnding the record in file is easy create a list structure and append in the file
#         f = open('shoe.dat', 'ab')
#         s_id = int(input("Enter Shoe ID : "))
#         s_name = input("Enter Shoe number : ")
#         s_brand = input("Enter Shoe Brand : ")
#         s_type = input("Enter Shoe type : ")
#         l = [s_id, s_name, s_brand, s_type]
#         pickle.dump(l, f)

#         f.close()
    
#     if ch == 2: # Displaying a CSV File
#         f = open('shoe.dat', 'rb')
#         while True:
#             try :
#                 dt = pickle.load(f)
#                 print("Here is your Record --> ", dt)
#             except EOFError:
#                 break
#         f.close()
    
#     if ch == 3: # Searching a Record
#         si = int(input("Enter Shoes ID: "))
#         f = open('shoe.dat', 'rb')
#         fl = False

#         while True:
#             try:
#                 dt= pickle.load(f)
#                 for i in dt:
#                     if i == si :
#                         fl = True
#                         print("Record Found ...")
#                         print(dt[0])
#                         print(dt[1])
#                         print(dt[2])
#                         print(dt[3])

#             except EOFError:
#                 break
        
#             if fl == False :
#                 print("Record Not found")
            
#         f.close()

#     if ch == 4 :
#         break


## CRUD Operation on CSV Files 
# Write a CRUD Operation in for a telephone.csv

def addRec():
    f = open('telephone.csv', 'a', newline='')
    wo = csv.writer(f)

    t_id = int(input("Enter the Id : "))
    t_name = input("Enter the name of the telephone : ")
    t_brand = input("Enter the brand of the telephobe : ")

    wo.writerow([t_id, t_name, t_brand])
    print("Record Added Succesfully")
    f.close()

def disRec():
    print("Displaying Records")
    f = open('telephone.csv', 'r')
    ro = csv.reader(f)
    l = list(ro)
    for i in range(1, len(l)):
        print("The record is here : ", l[i])
    f.close()

def seaRec():
    si = int(input("Enter the telephone ID : "))
    f = open('telephone.csv', 'r')
    ro = csv.reader(f)
    found = 0

    for i in ro :
        if si == i[0]:
            found = 1 
            print("Record found here it is : ", i)
    if found == 0: 
        print("Record Not found")

    f.close()

def menu():
    print("CSV crud operation")
    print("1. Add Record")
    print("2. Display Record")
    print("3. Search Record")
    
    ch = int(input("Enter your Desired selection: "))

    if ch == 1: 
        addRec()
    if ch == 2:
        disRec()
    if ch == 3:
        seaRec()


menu()