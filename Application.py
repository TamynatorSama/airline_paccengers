#libraries importation
import numpy as np
import streamlit as st
import pickle

#import gbc trained model
ada_model = pickle.load(open('./ada_model.sav', 'rb'))

#create interface
def Airline_Passenger_Pred():
    st.title('Airline Passenger Satisfaction Prediction Software')
    st.markdown('Kindly provide the inputs below')
    
    #build text input
    Gender = st.text_input('Gender of passenger - Male: 0, Female: 1')
    Customer_Type = st.text_input('Passenger Type - Loyal Customer: 0, disloyal customer: 1')
    Age = st.text_input('Age of passenger')
    Type_of_Travel = st.text_input('Type of Travel - Business: 0, Personal: 1')
    Class = st.text_input('Travelling class - Business: 0, Eco: 1, Eco Plus: 2')
    Flight_Distance = st.text_input('Flight distance')
    Inflight_wifi_service = st.text_input('Inflight wifi service')
    Departure_Arrival_time_convenient = st.text_input('Departure/arrival time convenient')
    Ease_of_Online_booking = st.text_input('Online booking')
    Gate_location = st.text_input('Gate location')
    Food_and_drink = st.text_input('Food and drink')
    Online_boarding = st.text_input('Online boarding')
    Seat_comfort = st.text_input('Seat comfort')
    Inflight_entertainment = st.text_input('Inflight entertainment')
    On_board_service = st.text_input('On-board service')
    Leg_room_service = st.text_input('Leg room service')
    Baggage_handling = st.text_input('Baggage handling0')
    Checkin_service = st.text_input('Checkin service')
    Inflight_service = st.text_input('Inflight service')
    Cleanliness = st.text_input('Cleanliness')
    Departure_Delay_in_Minutes = st.text_input('Departure  Delay in Minutes')
    Arrival_Delay_in_Minutes = st.text_input('Arrival Delay in Minutes')
    
    
    if st.button('Classify passenger'):
        #convert data into an array of data
        data = np.array([Gender, Customer_Type, Age, Type_of_Travel, Class, Flight_Distance, Inflight_wifi_service, 
                         Departure_Arrival_time_convenient, Ease_of_Online_booking, Gate_location, Food_and_drink, 
                         Online_boarding, Seat_comfort, Inflight_entertainment, On_board_service, Leg_room_service, 
                         Baggage_handling, Checkin_service, Inflight_service, Cleanliness, Departure_Delay_in_Minutes, 
                         Arrival_Delay_in_Minutes])
        
        #reshape the data to fit the model
        data_reshape = data.reshape([1, -1])
        
        #make prediction with the data
        result = ada_model.predict(data_reshape)
        result = int(result)
        
        if result == 0:
            st.info('This passenger will satisfy')
        else:
            st.success('This passenger is neutral or will dissasify')
        
        
        
if __name__ == '__main__':
    Airline_Passenger_Pred()