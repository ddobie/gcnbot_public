from twilio.rest import Client
from astropy.io import ascii
import time
import logging

def get_details(twilio_details):
  f = open(twilio_details,'rU')
  lines = f.readlines()

  account_sid = lines[0][:-1]
  auth_token = lines[1][:-1]
  outbound_number = lines[2][:-1]
  f.close()
  
  return account_sid, auth_token, outbound_number

def get_numbers(filename = 'phone_numbers.dat'):
  return ascii.read(filename)

def respond_to_alert(outbound_message, test=True):
  logging.info("SMS content: %s"%(outbound_message))
  account_sid, auth_token, outbound_number = get_details('twilio_details.dat')
  
  twilioclient = Client(account_sid, auth_token)
  
  if test:
    logging.info("Only sending SMS to my number")
    phone_numbers = get_numbers(filename='phone_numbers_testing.dat')
  else:
    logging.info("Send SMS to all recipients")
    phone_numbers = get_numbers(filename='phone_numbers.dat')
  
  for contact in phone_numbers:
    person = contact['name']
    number = contact['number']
    country = contact['country']
    receive = contact['receive']
    
    if receive != 'yes':
      continue
    
    if country == 'Australia':
      message = twilioclient.messages.create(
      to=number, 
      from_="sydgw",
      body=outbound_message)
    
    if country == 'USA':
      message = twilioclient.messages.create(
      to=number, 
      from_=outbound_number,
      body=outbound_message)
    
    logging.info("Sent SMS to %s"%(person))
      
    time.sleep(1.5) #Twilio has a 1 second cool-down
    
  return
  
if __name__ == '__main__':

  respond_to_alert('test message')
