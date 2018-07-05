import sys
from pyspark import SparkConf, SparkContext
#mllib library functions to do machine learning and predictive analysis
from pyspark.mllib.recommendation import ALS, Rating

# create a Spark context
conf = SparkConf().setMaster("local[*]").setAppName("BookRecommendationSystem")
sc = SparkContext(conf = conf)
sc.setCheckpointDir('checkpoint')

# from the books catalog file create a dictionary of bookID to name
def loadBookNamesDict():
    bookNames = {}
    with open("books.csv", encoding='ascii', errors="ignore") as file:
        for line in file:
            fields = line.split(',')
            bookID = int(fields[0])
            bookName = fields[10]
			#key value pair to map bookId to name
            bookNames[bookID] = bookName
    return bookNames

	#calls function to get back loaded dictionary and set it to the dict
print("Loading book names dictionary..")
bookNamesDict = loadBookNamesDict()

# now lets get to the ratings data
# lets create a RDD with Rating objects as expected by MLLib
ratings = sc.textFile("file:///SparkCourse/ratings.csv")
ratingsRDD = ratings.map(lambda l: l.split(',')) \
            .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))) \
            .cache()

# Build the recommendation model using Alternating Least Squares
print("Training recommendation model...")
#how many recommendations get printed
rank = 10
#more iterations get a more accurate result but also very time consuming
numIterations = 6
#define the ALS model
model = ALS.train(ratingsRDD, rank, numIterations)

# the user for which we need to recommend - for now it is hard coded
#userID = int(sys.argv[1])
userID = 49926

# lets print the ratings given by this user..
print("\nRatings given by userID " + str(userID) + ":")
userRatings = ratingsRDD.filter(lambda l: l[0] == userID)
#loop through the ratings and print the name (pass int he key of the dict) and it's rating
for rating in userRatings.collect():
    print (bookNamesDict[int(rating[1])] + ": " + str(rating[2]))

# now lets use our model to recommend books for this user..
print("\nTop 10 recommendations:")
recommendations = model.recommendProducts(userID, 10)
for recommendation in recommendations:
    print (bookNamesDict[int(recommendation[1])] + \
        " score " + str(recommendation[2]))
        
print('\n')