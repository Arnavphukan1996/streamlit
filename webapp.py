import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
  
# loading in the model to predict on the data
pickle_in = open('class1.pkl', 'rb')
classifier = pickle.load(pickle_in)



def welcome():
    return 'welcome all'
  
# defining the function which will make the prediction using 
# the data which the user inputs
def prediction(country,rating,duration,Genres,year_added,weekday):  
    prediction = classifier.predict([[country,rating,duration,Genres,year_added,weekday]])
    print(prediction)
    return prediction


# In[3]:


df=pd.read_csv("machine.csv")
df1=pd.read_csv("updatednetflix.csv")
st.sidebar.write("**About this Dataset:** Netflix is one of the most popular media and video streaming platforms. They have over 8000 movies or tv shows available on their platform, as of mid-2021, they have over 200M Subscribers globally. This tabular dataset consists of listings of all the movies and tv shows available on Netflix, along with details such as - cast, directors, ratings, release year, duration, etc.")
option_9 = st.sidebar.checkbox('Dataset Overview')
if option_9:
    def load_data():
        df=pd.read_csv("machine.csv")
        df1=df.drop(['Unnamed: 0','title','month_added','release_year','date_added'],axis=1)
        return df1
    def load1_data():
        df=pd.read_csv("updatednetflix.csv")
        df2=df.drop(['title','release_year','date_added'],axis=1)
        return df2

    st.sidebar.write("Which data you want to display?")

    #load original dataset
    data=load1_data()
    ch=st.sidebar.checkbox("Netflix Data")
    print(ch)
    if ch:
        st.write("Netflix Dataset")
        st.dataframe(data=data)
    #load machine dataset
    data=load_data()
    ch1=st.sidebar.checkbox("Machine Generated Data")
    print(ch1)
    if ch1:
        st.write("Machine Generated Dataset")
        st.dataframe(data=data)



# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    #st.title("Netflix Show Prediction")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:powderblue;padding:13px">
    <h1 style ="color:black;text-align:center;">Netflix Show Recommender App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
   
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    country = st.selectbox("Country", df['country'].tolist())
    rating = st.selectbox("Rating",df['rating'].tolist())
    duration = st.selectbox("Duration",df['duration'].tolist())
    Genres = st.selectbox("Genres",df['Genres'].tolist())
    year_added = st.selectbox("Year",df['year_added'].tolist())
    weekday = st.selectbox("Week",df['weekday'].tolist())
    
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(country,rating,duration,Genres,year_added,weekday)
        if result:
            st.success('Model Recommended it as a  {}'.format(result))
    
        
    
    st.sidebar.write("**Interesting Task Ideas:**") 
    
    option_8 = st.sidebar.checkbox('Exploratory Data Analysis')
    if option_8:
        st.write("**EDA**")

        st.sidebar.write('Netflix Shows Based on Country')
        option_1 = st.sidebar.checkbox('TopFive')
        option_2 =st.sidebar.checkbox('BottomFive')

        st.sidebar.write("How do you want to see the trends on Netflix?")
        option_3 = st.sidebar.checkbox('Year Wise')
        option_4 = st.sidebar.checkbox('Month Wise')
        option_5 = st.sidebar.checkbox('Date Wise')
        option_6 = st.sidebar.checkbox('Week Wise')
        option_7 = st.sidebar.checkbox('Day Wise')

        if option_1:
            st.write('Top Five Countries Producing Netflix Shows')
            #top 5 country with number of shows
            topfive=df1['country'].value_counts()[:5]
            x1= topfive.index
            y1= topfive.values
            plt.figure(figsize=(12,6))
            fig1=px.bar(df1,x=x1,y=y1)
            fig1.update_layout(xaxis_title="Country",yaxis_title="Number of shows")
            st.write(fig1)
            with st.expander("Click here more details"):
                st.write("""The chart above shows some numbers I picked for you.I rolled actual dice for these, so they're *guaranteed* to be random.""")
        if option_2:

            st.write('Bottom Five Countries Producing Netflix Shows')
            #bottom 5 country with number of shows
            lastfive=df1['country'].value_counts().tail()
            x1= lastfive.index
            y1= lastfive.values
            plt.figure(figsize=(12,6))
            fig2=px.bar(df1,x=x1,y=y1)
            fig2.update_layout(xaxis_title="Country",yaxis_title="Number of shows")
            st.write(fig2)

        if option_3:
            st.write('Year wise Producing Netflix Shows')
            #displaying the number of shows released yearwise
            x1=df1[df1['year_added']>2015]
            x=x1['year_added'].unique()
            #x=df1['year_added'].unique()
            y=x1['year_added'].value_counts()
            plt.figure(figsize = [10,5])
            fig3=px.bar(df1,x=x,y=y)
            fig3.update_layout(xaxis_title="Year",yaxis_title="Number of shows")
            st.write(fig3)

        if option_4:
            df1['month_added'].replace({1:'January', 
                                2:'February',
                                3:'March',
                                4:'April',
                                5:'May',
                                6:'June',
                                7:'July',
                                8:'August',
                                9:'September',
                                10:'October',
                                11:'November',
                                12:'December'},inplace=True)
            st.write('Month wise Producing Netflix Shows')
            #displaying the number of shows released monthwise
            x=df1['month_added'].unique()
            y=df1['month_added'].value_counts().sort_values(ascending=True)
            plt.figure(figsize = [10,5])
            plt.xticks(rotation =45)
            fig4=px.bar(df1,x=x,y=y)
            fig4.update_layout(xaxis_title="Month",yaxis_title="Number of shows")
            st.write(fig4)

        if option_5:
            st.write('Date wise Producing Netflix Shows')
            #displaying the number of shows released daywise
            x=df1['date_added'].unique()
            y=df1['date_added'].value_counts()
            plt.figure(figsize = [10,6])
            plt.xticks(rotation =45)
            fig5=px.bar(df1,x=x,y=y)
            fig5.update_layout(xaxis_title="Date",yaxis_title="Number of shows")
            st.write(fig5)

        if option_6:
            st.write('Week wise Producing Netflix Shows')
            #highest weekwise shows
            week1=df1[(df1['date_added']>=1) & (df1['date_added']<8)]
            week2=df1[(df1['date_added']>=8) & (df1['date_added']<15)]
            week3=df1[(df1['date_added']>=15) & (df1['date_added']<22)]
            week4=df1[(df1['date_added']>=22) & (df1['date_added']<31)]

            xweek = ['week 1','week 2','week 3','week 4']
            yweek = [len(week1), len(week2), len(week3), len(week4)]
            fig6=px.bar(df1,x=xweek,y=yweek)
            fig6.update_layout(xaxis_title="Week",yaxis_title="Number of shows")
            st.write(fig6)

        if option_7:
            st.write('Day wise Producing Netflix Shows')
            def days(weekday):
                if weekday<2:
                    return 'Monday'
                elif weekday<3:
                    return 'Tuesday'
                elif weekday<4:
                    return 'Wednesday'
                elif weekday<5:
                    return 'Thursday'
                elif weekday<6:
                    return 'Friday'
                else:
                    return 'Saturday'
            df1['weekday']=df1['weekday'].apply(days)
            p=df1['weekday'].value_counts().sort_values(ascending=True)
            x=p.keys()
            y=p.values
            fig7=px.bar(df1,x=x,y=y)
            fig7.update_layout(xaxis_title="Week",yaxis_title="Number of shows")
            st.write(fig7)


    
if __name__=='__main__':
    main()


# In[ ]:




