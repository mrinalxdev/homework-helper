## All the Crud Operation in One File
import pickle
## 1. Binary File Crud Operation
## Write a Program for crud operation for a shoe store management


while True:
    print("1. Add Record ")
    print("2. Display Record ")
    print("3. Search Record ")
    print("4. Exit")

    ch = int(input("Enter your operation : "))
    l = []

    if ch == 1: # Appnding the record in file is easy create a list structure and append in the file
        f = open('shoe.dat', 'ab')
        s_id = int(input("Enter Shoe ID : "))
        s_name = input("Enter Shoe number : ")
        s_brand = input("Enter Shoe Brand : ")
        s_type = input("Enter Shoe type : ")
        l = [s_id, s_name, s_brand, s_type]
        pickle.dump(l, f)

        f.close()
    
    if ch == 2: # Displaying a CSV File
        f = open('shoe.dat', 'rb')
        while True:
            try :
                dt = pickle.load(f)
                print("Here is your Record --> ", dt)
            except EOFError:
                break
        f.close()
    
    if ch == 3: # Searching a Record
        si = int(input("Enter Shoes ID: "))
        f = open('shoe.dat', 'rb')
        fl = False

        while True:
            try:
                dt= pickle.load(f)
                for i in dt:
                    if i == si :
                        fl = True
                        print("Record Found ...")
                        print(dt[0])
                        print(dt[1])
                        print(dt[2])
                        print(dt[3])

            except EOFError:
                break
        
            if fl == False :
                print("Record Not found")
            
        f.close()

    if ch == 4 :
        break


## CRUD Operation on CSV Files 

