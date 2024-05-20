# this file reads in key_words_dict.pkl and uses it as a knowledge base to inform the chat bot

import os
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
nltk.download('punkt')
import string
import pickle
import random
from aeg_naive_bayes_classifier import NaiveBayesClassifier, preprocess, feature_extraction, load_data_from_csv # import my aeg_naive_bayes_classifier


has_traveled = False
has_liked_topic = False
has_liked_city = False

# Get the directory of the current script
dir_path_curr_script = os.path.dirname(os.path.realpath(__file__))
users_folder_path = os.path.join(dir_path_curr_script,'users')
os.makedirs(users_folder_path, exist_ok=True) #make sure the folder exists

pickle_file_path = os.path.join(dir_path_curr_script, 'key_words_dict.pkl')

key_words_pickle_dict = {}
with open(pickle_file_path, 'rb') as file:
    key_words_pickle_dict = pickle.load(file)

#stores user information - likes and dislikes, and travel history
user_dict = \
{ 
    "name":"",
    "has-been-to":[],
    "hasnt-been-to":[],
    "likes-cities":[],
    "dislikes-cities":[],
    "likes-topics":[],
    "dislikes-topics":[],
}

#function to list capabilities specifically, so that a grader can learn quickly, or someone without access to documentation can see
def list_capabilities(place_holder_token): #place holder intentionally does nothing - but other functions called the same way need an input there so it makes sense to have a placeholder
    place_holder = place_holder_token
    place_holder + ""
    print('''
[^_^] ROB: Here's a more specific and thorough overview of my abilities and limitations.<[o.o]>
1. CITY FEATURES: You can ask me about these travel topics: beaches, mountains, music, parties, museums, restaurants, or bars. For any of these topics, I mention a US city or two that I think has a good 
scene for that type of fun. I know a lot [^_^], so if you ask multiple times you'll probably get different answers.\n
2. *GENERAL*, OPEN-ENDED RECOMMENDATION: You can ask me to pick entirely on my own, and recommend a city and an activity together. Key word: GENERAL. [^_^]\n
3. ASK ABOUT A SPECIFIC CITY: You can also ask me to recommend positive features and activities of a specific city you name [^_^]
I can recommend dozens of things about these cities: Dallas, Austin, New York, Los Angeles, Las Vegas, New Orleans, San Antonio, Houston, Washington DC, 
Orlando, Miami, Denver, Chicago, Boston, San Francisco, and Portland.
--- If you ask about a specific city, I will always reply about it, but you cannot request a specific topic for the reply. [o.o] Sorry :|
''')

# function to add user's US travel history
def add_history(matching_place_holder_token): #place holder intentionally does nothing - but other functions called the same way need an input there so it makes sense to have a placeholder
    place_holder = matching_place_holder_token
    place_holder + ""
    print("")

# function to recommend travel destinations for any non-Austin city
def city_func(matching_city_token):
    print("[^_^] ROB:"+current_first_name_choice,random.choice(key_words_pickle_dict[matching_key_word_to_pickled_dict[matching_city_token]]),current_transition,
          random.choice(key_words_pickle_dict[matching_key_word_to_pickled_dict[matching_city_token]]))
    ask_about_travel_history(matching_city_token)

# function to recommend travel destinations based on topic, not city
def topic_func(matching_topic_token):
    print("[^_^] ROB:"+current_first_name_choice, random.choice(key_words_pickle_dict[matching_key_word_to_pickled_dict[matching_topic_token]]),current_transition,
          random.choice(key_words_pickle_dict[matching_key_word_to_pickled_dict[matching_topic_token]]))
    ask_about_activity_likes(matching_topic_token)

# function to recommend a city and topic, without input from the user
def general_func(matching_place_holder_token): #place holder intentionally does nothing - but other functions called the same way need an input there so it makes sense to have a placeholder
    place_holder = matching_place_holder_token
    place_holder + ""
    attempts = 0
    while attempts < 25:
        try:
            print("[^_^] ROB:"+current_first_name_choice, random.choice(random.choice(list(key_words_pickle_dict.values()))),
                random.choice(random.choice(list(key_words_pickle_dict.values()))),
                random.choice(random.choice(list(key_words_pickle_dict.values()))))
            break
        except:
            attempts += 1

# function to ask if they like/dislike an activity such as mountains, music, bars, or restaurants - and remember their interests!
def ask_about_activity_likes(activity_string):
    if activity_string in user_dict["likes-topics"] or activity_string in user_dict["dislikes-topics"]:
        return
    user_input = input(f"\n[^_^] ROB: Do you like {output_topics_dict_plural[activity_string]}? (type your answer and press enter)\n\n")
    #print(f"You typed: {user_input}")
    input_tokens = word_tokenize(user_input.lower())
    #stop words - keep stop words since "no" is a stopword
    input_tokens = [token for token in input_tokens if token.isalpha()]
    
    input_tokens = [lemmatizer.lemmatize(input_token) for input_token in input_tokens]
    for input_token in input_tokens:
        if any(token in ['yes','yeah',] for token in input_tokens):
            user_dict["likes-topics"].append(activity_string)
            global has_liked_topic
            has_liked_topic = True
            break
        elif any(token in ['no','nope','nah',] for token in input_tokens):
            user_dict["dislikes-topics"].append(activity_string)
            break

# function to ask if a user has been to a specific city - and if they have, ask + remember if they like/dislike it!     
def ask_about_travel_history(city_string):
    if city_string in user_dict["has-been-to"] or city_string in user_dict["hasnt-been-to"]:
        return
    user_input = input(f"\n[^_^] ROB: Have you ever been to {output_cities_dict[city_string]}? (type your answer and press enter)\n\n")
    #print(f"You typed: {user_input}")
    input_tokens = word_tokenize(user_input.lower())
    #stop words - keep stop words since "no" is a stopword
    input_tokens = [token for token in input_tokens if token.isalpha()]
    
    input_tokens = [lemmatizer.lemmatize(input_token) for input_token in input_tokens]
    for input_token in input_tokens:
        if any(token in ['yes','yeah',] for token in input_tokens):
            global has_traveled
            has_traveled = True
            user_dict["has-been-to"].append(city_string)
            user_input1 = input(f"\n[^_^] ROB: Oh wow! Did you like {output_cities_dict[city_string]}? (type your answer and press enter)\n\n")
            #print(f"You typed: {user_input}")
            input_tokens1 = word_tokenize(user_input1.lower())
            #stop words - keep stop words since "no" is a stopword
            input_tokens1 = [token for token in input_tokens1 if token.isalpha()]
            input_tokens1 = [lemmatizer.lemmatize(input_token) for input_token in input_tokens1]
            for input_token in input_tokens1:
                if any(token in ['yes','yeah',] for token in input_tokens):
                    global has_liked_city
                    has_liked_city = True
                    user_dict["likes-cities"].append(city_string)
                    
                    break
                elif any(token in ['no','nope','nah',] for token in input_tokens):
                    user_dict["dislikes-cities"].append(city_string)
                    break
            break
        elif any(token in ['no','nope','nah',] for token in input_tokens):
            user_dict["hasnt-been-to"].append(city_string)
            break
    

key_word_tokens = \
[
    'list','capability','capabilities',
    'austin', 'dallas', 'ny', 'nyc', 'york', 'la', 'angeles', 'vega', 
    'orleans', 'antonio', 'houston','washington','dc',
    'orlando', 'miami', 'denver', 'chicago', 'boston', 'francisco', 'portland',  
    'beach', 'mountain', 'music','party','museum','restaurant', 'bar',
    'general','recommendation','recommend','suggest','suggestion',
]

# this dict translates the key_word_token that matches the user_input into the appropriate action
key_word_functions_dict = \
{
    'list': list_capabilities, 'capability': list_capabilities,'capabilities': list_capabilities,
    'austin': city_func, 'dallas': city_func, 'ny': city_func, 'nyc': city_func, 'york': city_func, 
    'la': city_func, 'angeles': city_func, 'vega':city_func, 
    'orleans':city_func, 'antonio':city_func, 'houston':city_func,'washington':city_func,'dc':city_func,
    'orlando':city_func, 'miami':city_func, 'denver':city_func, 'chicago':city_func, 'boston':city_func, 'francisco':city_func, 'portland':city_func,  
    'beach':topic_func, 'mountain':topic_func, 'music':topic_func,'party':topic_func,'museum':topic_func,'restaurant':topic_func, 'bar':topic_func,
    'general':general_func,'recommendation':general_func,'recommend':general_func,'suggest':general_func,'suggestion':general_func,
}

# this dict translates the key_word_token that matches user_input into the appropriate key to find quotes in the pickled dict
# the pickled dict keys had to work for searching a corpus, which is a slightly different requirement set than the key_words that match the user input
# often this dict matches other dicts, but not always
matching_key_word_to_pickled_dict = \
{
    'austin':'Austin', 'dallas':'Dallas', 'ny':'York', 'nyc':'York', 'york':'York', 'la':'Angeles', 'angeles':'Angeles', 'vega':'Vegas', 
    'orleans':'Orleans', 'antonio':'San Antonio', 'houston':'Houston','washington':'D. C.','dc':'D. C.',
    'orlando':'Orlando', 'miami':'Miami', 'denver':'Denver', 'chicago':'Chicago', 'boston':'Boston', 'francisco':'Francisco', 'portland':'Portland',
    'beach':'beach', 'mountain':'mountain', 'music':'music', 'party':'party', 'museum':'museum', 'restaurant':'restaurant', 'bar':'bar',
}

# OUTPUT
# this dict translates the key_word that matches the user's input into the appropriate form to print the city name
output_cities_dict = \
{
    'austin':'Austin', 'dallas':'Dallas', 'ny':'New York', 'nyc':'New York', 'york':'New York', 'la':'Los Angeles', 'angeles':'Los Angeles', 'vega':'Las Vegas', 
    'orleans':'New Orleans','antonio':'San Antonio', 'houston':'Houston', 'washington':'Washington DC','dc':'Washington DC',
    'orlando':'Orlando', 'miami':'Miami', 'denver':'Denver', 'chicago':'Chicago', 'boston':'Boston', 'francisco':'San Francisco',  'portland':'Portland',  
}

#translates key matching user input to appropriate output format
output_topics_dict = {'beach':'beach', 'mountain':'mountain', 'music':'music', 'party':'party', 'museum':'museum', 'restaurant':'restaurant', 'bar':'bar',} 
output_topics_dict_plural = {'beach':'beaches', 'mountain':'mountains',  'music':'different types of music', 'party':'parties', 'museum':'museums', 'restaurant':'restaurants', 'bar':'bars',}

# "jokes" and funny comments that appear at times that will appear random to a first time user
jokes_dict = \
{1:"[^_^] ROB: I'm a computer program [o.o], so I've never actually been to any of the places I talk about <[o.o]> ...\
but I sure know a lot about them! [^_^]", 2:"[^_^] ROB: I like you. I'll put in a good word for you when AI takes over the world :P",
3:"[-.-] ROB: ZZZ...ZZZ...ZZZ [-.-] || [*_*!!] WHOA WHOA OK OK I'M UP! [>.>]Sorry Robots don't go to sleep because they're bored...[<.<] I just needed to recharge \
my batteries...[=.=] I was dreaming of electric sheep...[>.<]", 4:"[^_^] ROB: You know, I think you might like me. That's fine, I'm a very loveable \
little Robot! [•_-] Beep boop [^_^]"}


# first thing bot says - request user name
print("[^_^] ROB: Hi, I'm ROB the Robot, your own personal travel guide!")
user_input = full_name = input("[^_^] ROB: Please type your name and press enter: ")
first_name = full_name.split()[0]
#print(f"You typed: {user_input}")
user_dict["name"] = user_input
user_file_name = user_input+'.txt'
user_file_path = os.path.join(users_folder_path, user_input)

with open(user_file_path, 'w', encoding = 'utf-8') as output_file:
    json.dump(user_dict, output_file, ensure_ascii=False, indent = 4)


# bot's opening message to USER NAME
print('''
[^_^] ROB: Nice to meet you,''',first_name+".",'''I'm here to help you have the time of your life in US cities! I'm a travel guide to over 15 US cities. 
[^_^] ROB: I know over 700 pieces of information curated from over 550 urls in my knowledge base! Not bad for a school project, huh?
[^_^] ROB:       
1. CITY FEATURES: Ask me for a city that is known for it's good mountains, beaches, restaurants, bars, music, or more [^_^]
2. REQUEST A *GENERAL* RECOMMENDATION: You can also ask me choose for you - a general recommendation of where you should go next and what you should do there.
3. ASK ABOUT A SPECIFIC CITY: If you have a US tourist city in mind already, I can give you travel tips for when you get there [^_^]
4. LIST OF *CAPABILITIES*: You can also request a more specific list of my capabilities (this may help with an evaluation [-_-]).''')


# key pre-loop initializations
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
first_try = False
user_prompt_string = ""
joke_count = 0
custom_comment_count = 0

filename = 'Synthetic_Request_Data.csv'
data = load_data_from_csv(filename)
NB_classifier = NaiveBayesClassifier()
NB_classifier.train(data)



# post-greeting loop that runs while the bot is active until terminal is closed
while True:

    # before user input, a new transition word is chosen to keep things fresh
    with open(user_file_path, 'w', encoding = 'utf-8') as output_file:
        json.dump(user_dict, output_file, ensure_ascii=False, indent = 4)
    current_first_name_choice = random.choice([" "+first_name+",","",""])
    current_transition = random.choice(["Also,","Also,","Plus,","Plus","Furthermore,","Furthermore,", "To add to that,","On top of that,","But there's more:","Additionally,","And get this:"])
    if joke_count == 2:
        print(jokes_dict[1])
    elif joke_count == 4:
        print(jokes_dict[2])
    elif joke_count == 6:
        print(jokes_dict[3])
    elif joke_count == 9:
        print(jokes_dict[4])

    # BEFORE USER INPUT - CUSTOM COMMENT - CHANGES BASED ON WHETHER THE USER HAS SPECIFIED THEY LIKED ANY CITIES OR TRAVEL ACTIVITIES
    if custom_comment_count == 2:
        #print(f"has-liked-city:{has_liked_city}")
        #print(f"has_liked_topic:{has_liked_topic}")
        #print(f"has_traveled:{has_traveled}")
        custom_comment_count = 0
        if has_liked_city == True and has_liked_topic == True:
            random_city_like_v1 = output_cities_dict[random.choice(user_dict["likes-cities"])]
            random_topic_like = output_topics_dict_plural[random.choice(user_dict["likes-topics"])]
            print(random.choice([
                f"[^_^] ROB: I hope I can show you more cities that you like as much as {random_city_like_v1}!"
                f"[^_^] ROB: I hope I can show you more places where you can enjoy {random_topic_like}!"
            ]))
        elif has_liked_city == True:
            random_city_like_v1 = output_cities_dict[random.choice(user_dict["likes-cities"])]
            print(f"[^_^] ROB: I hope I can show you more cities that you like as much as {random_city_like_v1}!")
        elif has_liked_topic == True:
            random_topic_like = output_topics_dict_plural[random.choice(user_dict["likes-topics"])]
            print(f"[^_^] ROB: I hope I can show you more places where you can enjoy {random_topic_like}!")
        else:
            print("[^_^] ROB: I hope I can show you something that you like!")

    # PRIMARY SET OF PROMPTS FOR USER INPUT
    if first_try == False:
        user_prompt_string = "\n[^_^] ROB: Ask me a travel question in natural language and press enter...\n\n"
    else:
        user_prompt_string = random.choice([
            "\n[^_^] ROB: Ask me another travel question in natural language and press enter...\n\n"
            "\n[^_^] ROB: Do you have another question? Ask away, then press enter...\n\n"
            "\n[^_^] ROB: I'm here to answer your questions. Just type what's on your mind, then press enter...\n\n"          
            "\n[^_^] ROB: Just type what you want to know, then press enter...\n\n" 
        ])

    # receive user input
    user_input = input(user_prompt_string)
    first_match_token = ""
    first_match_token = NB_classifier.predict(user_input)
    #print(f"The text is classified as: {first_match_token}")
    if first_match_token == "":
        first_match_token = "general"
    matching_function = key_word_functions_dict.get(first_match_token)
    matching_function(first_match_token)
    #print(f'This is what I saw:',user_input )

    # iterate to keep jokes and custom comments about past-metioned likes fresh and different
    joke_count += 1
    if joke_count > 20:
        joke_count = 0
    custom_comment_count += 1