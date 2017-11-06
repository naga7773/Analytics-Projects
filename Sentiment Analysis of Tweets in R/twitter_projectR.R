#install.packages("twitteR")
#install.packages("ROAuth")
#install.packages("wordcloud")
#install.packages("tm")
#install.packages("SnowballC")
#install.packages("tidytext")
#install.packages("lubridate)
#install.packages("gutenbergr")
rm(list = ls())

library(twitteR)
library(ROAuth)
library(wordcloud)
library(tm)
library(SnowballC)
library(tidytext)
library(lubridate)
library(dplyr)
library(gutenbergr)

setup_twitter_oauth(consumer_key = "a4AIBLHi7vgIg6fV1LczWHbf1",
                    consumer_secret = "XqjOsQUgdRiszXgiWaHoisd1TvC8AIF0zBPf22zqIKsQ7hFYLJ",
                    access_token = "562821993-hls8J3bj4mkuVMwmidksCxtxVYKfDoE7irw4y1Bb",
                    access_secret = "CQulousHrasSfCePGOwsHP85oNpEQCZfvYSF0dMaIeHtR")







#necessary file in windows
#download.file(url="http://curl.haxx.se/ca/cacert.pem", destfile="cacert.pem")
# '#iphone8' + 'iphone'

iphoneTweets <- searchTwitter('#HandsOffMyBC',
                              since='2017-09-11',
                              lang = "en",
                              n = 10000)

BirthControlTweets <- iphoneTweets
#r_stats <- searchTwitter("#WorldSmileDay", n=10000)   
iphoneTweets_text <- sapply(iphoneTweets, function(x) x$getText())

# RemoveSpecialCharacters <- function(x){
#   gsub("[[:punct:]]","",x)
# }
# r_stats_text_cleaned <- lapply(r_stats_text,RemoveSpecialCharacters)

r_stats_text_testiconv <- lapply(iphoneTweets_text,
                                 function(row) iconv(row,
                                                     "latin1", "ASCII", sub=""))



r_stats_text_corpus <- Corpus(VectorSource(r_stats_text_testiconv))

r_stats_text_corpus <- tm_map(r_stats_text_corpus, content_transformer(tolower)) 




r_stats_text_corpus <- tm_map(r_stats_text_corpus, removePunctuation)
r_stats_text_corpus <- tm_map(r_stats_text_corpus, function(x)removeWords(x,stopwords()))
#wordcloud(r_stats_text_corpus)


pal2 <- brewer.pal(8,"Dark2")
wordcloud(r_stats_text_corpus,min.freq=4,max.words=20, random.order=T, colors=pal2)






df1 <- twListToDF(iphoneTweets)


summary(df1$retweetCount)


Me$location

dtm <- DocumentTermMatrix(r_stats_text_corpus)

b<-colSums(as.matrix(dtm))

length(b)

b_ordered <- order(b, decreasing = T)

dtm_matrix <- as.matrix(dtm)


dtm_matrix[0:3]


t <- iphoneTweets_text[1:3]

g <- grepl(":",t[1])



which(t == "R")

x <-t[1]

grep(":",x,value = F)

typeof(x)

df1[1:3,]

twb <- tibble::as_tibble(df1)

TweetText <- twb%>%
  select(text)
Tweets <- as.vector(df1[,1])





bing <- get_sentiments("bing")

cc <- Tweets[1:10]

cc <- unnest_tokens(twb,word,text)

nrc <- get_sentiments("nrc")




nrc_analysis <- cc%>%
  select(word)%>%
  inner_join(nrc,by = "word")

bing_analysis <- cc%>%
  select(word)%>%
  inner_join(bing,by = "word")

bing_analysis%>%
  count(word)%>%
  arrange(desc(n))

e <- as.data.frame(word = c(1:10),sentiment = c(1:10))
bing_analysis$sentiment <- as.factor(bing_analysis$sentiment)
bing_analysis$word <- as.factor(bing_analysis$word)




ACD_works <- gutenberg_metadata %>%
  filter(author == "Doyle, Arthur Conan",has_text == T)



# ACD_metadata <- gutenberg_metadata %>%
#   filter(author == "Doyle, Arthur Conan",
#          language == "en",
#          has_text == T,
#          !str_detect(rights, "Copyright")) 

ACD_text_221 <- gutenberg_download(gutenberg_id =221)

ACD_text_221  <- ACD_text_221 %>%
  mutate(title = "The Return of Sherlock Holmes",
         linenumber = row_number())

ACD_text_126 <- gutenberg_download(gutenberg_id = 126)%>%
  mutate(title = "The Poision Belt",
         linenumber = row_number())


ACD_text_903 <- gutenberg_download(gutenberg_id = 903)%>%
  mutate(title = "The White Company",
         linenumber = row_number())

ACD_three_works <- bind_rows(ACD_text_126,ACD_text_221,ACD_text_903)


ACD_tokens <- ACD_three_works%>%
  unnest_tokens(word,text)%>%
  anti_join(stop_words,by = "word")%>%
  inner_join(get_sentiments("afinn"),by = "word")


# nrc <- get_sentiments("nrc")
# 
# afinn <- get_sentiments("afinn")


ACD_tokens <- ACD_tokens%>%
  mutate(Block = ceiling(linenumber / 70))%>%
  group_by(gutenberg_id,title,Block)%>%
  mutate(NoofWords = count())





TweetTibble <- tibble::as_tibble(TweetText)


Tweets <- TweetTibble%>%
  filter(text)%>%
  mutate(TweetNumber = rownumber())%>%
  unnest_tokens(text)







