##Basic MPN
#train on 14v2, test on rest
python train.py
python train.py --testcase 118v2
python train.py --testcase 6470rtev2

#train on 118v2, test on rest
python train.py --case 118v2
python train.py --case 118v2 --testcase 118v2
python train.py --case 118v2 --testcase 6470rtev2

#train on 6470rtev2, test on rest
python train.py --case 6470rtev2
python train.py --case 6470rtev2 --testcase 118v2
python train.py --case 6470rtev2 --testcase 6470rtev2


##MPN with No Message Passing
#train on 14v2, test on rest
python train_No.py
python train_No.py --testcase 118v2
python train_No.py --testcase 6470rtev2

#train on 118v2, test on rest
python train_No.py --case 118v2
python train_No.py --case 118v2 --testcase 118v2
python train_No.py --case 118v2 --testcase 6470rtev2

#train on 6470rtev2, test on rest
python train_No.py --case 6470rtev2
python train_No.py --case 6470rtev2 --testcase 118v2
python train_No.py --case 6470rtev2 --testcase 6470rtev2



##MPN with One layer
#train on 14v2, test on rest
python train_One.py
python train_One.py --testcase 118v2
python train_One.py --testcase 6470rtev2

#train on 118v2, test on rest
python train_One.py --case 118v2
python train_One.py --case 118v2 --testcase 118v2
python train_One.py --case 118v2 --testcase 6470rtev2

#train on 6470rtev2, test on rest
python train_One.py --case 6470rtev2
python train_One.py --case 6470rtev2 --testcase 118v2
python train_One.py --case 6470rtev2 --testcase 6470rtev2


##MPN with One layer and no message passing
#train on 14v2, test on rest
python train_None.py
python train_None.py --testcase 118v2
python train_None.py --testcase 6470rtev2

#train on 118v2, test on rest
python train_None.py --case 118v2
python train_None.py --case 118v2 --testcase 118v2
python train_None.py --case 118v2 --testcase 6470rtev2

#train on 6470rtev2, test on rest
python train_None.py --case 6470rtev2
python train_None.py --case 6470rtev2 --testcase 118v2
python train_None.py --case 6470rtev2 --testcase 6470rtev2