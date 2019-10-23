import os
import time
import re
from slackclient import SlackClient
import urllib.request

# Python standard library imports
import tempfile
import shutil
import sys
import glob
import logging
import logging.handlers
import logging.config
import string
import smtplib
import time


# Third-party imports
import gcn
import gcn.handlers
import gcn.notice_types
import requests
import numpy as np
import healpy as hp

import astropy.utils.data
import lxml.etree

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

from paramiko import SSHClient, SSHConfig
import paramiko
from scp import SCPClient

import matplotlib as mpl
mpl.use('Agg')

# My own scripts
import observability
import send_SMS

# Logging setup
class BufferingSMTPHandler(logging.handlers.BufferingHandler):
  def __init__(self, mailhost, fromaddr, toaddrs, subject, capacity):
    logging.handlers.BufferingHandler.__init__(self, capacity)
    self.mailhost = mailhost[0]
    self.mailport = mailhost[1]
    self.fromaddr = fromaddr
    self.toaddrs = toaddrs
    self.subject = subject
    self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s"))
    self.lastflush = Time.now()

  def flush(self):
    if len(self.buffer) > 0:
      try:
        port = self.mailport
        smtp = smtplib.SMTP(self.mailhost, port)
        msg = "From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n" % (self.fromaddr, ",".join(self.toaddrs), self.subject)
        for record in self.buffer:
          s = self.format(record)
          msg = msg + s + "\r\n"
        smtp.sendmail(self.fromaddr, self.toaddrs, msg)
        smtp.quit()
      except:
        self.handleError(None)  # no particular record
      #self.buffer = []
      super(BufferingSMTPHandler, self).flush()
      
  def shouldFlush(self, record):
    levelno = record.levelno
    
    if levelno > 30: # respond to errors and criticals
      
      #Check for a gcn listen timeout error
      if record.msg == 'could not connect to %s:%d, will retry in %d seconds':
        num_secs = record.args[2]
        if num_secs < 300:
          return False
      
      return True
    
    return False
    
  def close(self):
    logging.Handler.close(self)
    
    
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('gcnbot.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s"))

smtp_handler = BufferingSMTPHandler(mailhost=("smtp.sydney.edu.au", 25),
                                            fromaddr="ddob1600@uni.sydney.edu.au", 
                                            toaddrs=["ddobie94@gmail.com"],
                                            subject="CRITICAL: gcnbot logging",
                                            capacity=20)
smtp_handler.setLevel(logging.DEBUG)

logger.addHandler(smtp_handler)
logger.addHandler(file_handler)

def get_max_position(skymap):
  npix = len(skymap)
  nside = hp.npix2nside(npix)
  
  pix_max = np.argmax(skymap)
  
  theta, phi = hp.pix2ang(nside, pix_max)
  ra = np.rad2deg(phi)
  dec =  np.rad2deg(0.5*np.pi-theta)
  
  return SkyCoord(ra, dec, unit='deg')
  
def calculate_localisation(skymap,localisation=0.9):
  npix = len(skymap)
  sky_area = 4 * 180**2 / np.pi
  pix_size = sky_area/npix
  

  skymap_cumsum = np.cumsum(np.sort(skymap)[::-1])
  localisation_npix = np.argmax(skymap_cumsum >= localisation)
  
  localisation_size = localisation_npix * pix_size
  
  return localisation_size

def skymap_strings(params):

  skymap_loaded = False
  
  i = 0
  while not skymap_loaded:
    try:
      # Read the HEALPix sky map and the FITS header.
      skymap, header = hp.read_map(params['skymap_fits'], h=True, verbose=False)
      skymap_loaded = True
    except:
      logger.info('Skymap unavailable: waiting 10 seconds')
      time.sleep(10)
      i += 1
      
    if i > 10:
      logger.info('Skymap unavailable: giving up')
      return '\n Error: Skymap not available','',''
      
  header = dict(header)

  dist_txt = "Distance: %.1f +/- %.1f Mpc"%(header['DISTMEAN'], header['DISTSTD'])

  coord_max = get_max_position(skymap)
  
  current_time = astropy.time.Time.now()
  sun_pos = astropy.coordinates.get_sun(current_time)
  solar_angle = sun_pos.separation(coord_max)
  
  coord_max_string = coord_max.to_string(style='hmsdms')
  pos_txt = "Posterior max at: %s (%.1fdeg from Sun)"%(coord_max_string, solar_angle.degree)
  
  localisation_size = calculate_localisation(skymap)
  localisation_txt = "90%% localisation: %.1f sq.deg."%(localisation_size)
  
  return dist_txt, pos_txt, localisation_txt


  
  #Paranal = EarthLocation(lat=-24.6272*u.deg, lon=-70.4039*u.deg, height=100*u.m)
  
  
  
def copy_file(infile, outfile, target_host='gateway'):
  logger.debug('Copying %s to %s:%s'%(infile, target_host, outfile))
  ssh = SSHClient()
  
  ssh_config_file = os.path.expanduser("~/.ssh/config")
  
  hostname = None
  if os.path.exists(ssh_config_file):
    conf = SSHConfig()
    
    with open(ssh_config_file) as f:
      conf.parse(f)
    
    host_config = conf.lookup(target_host)
    hostname = host_config['hostname']
  
  ssh.load_system_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
  ssh.connect(hostname)

  #ssh.connect(target_host)
  
  with SCPClient(ssh.get_transport()) as scp:
    scp.put(infile, outfile)
    scp.close()

  return

def make_observability_plot(params):
  skymap_full, header = hp.read_map(params['skymap_fits'], h=True, verbose=False)
  header = dict(header)
  
  skymap = skymap_full
  skymap = observability.downsample_skymap(skymap_full)
  
  obs_plot_fname = "obs_plots/%s_observability.png"%(params['GraceID'])
  
  observability.observable_probabilty_plot(skymap, filename=obs_plot_fname)
  
  copy_file(obs_plot_fname,'/import/www/personal/ddob1600/gcnbot/%s'%(obs_plot_fname))
  
  image_url = "http://www.physics.usyd.edu.au/~ddob1600/gcnbot/%s"%(obs_plot_fname)
  
  attachments = [{"title": "%s observability"%(params['GraceID']), "image_url": image_url}]
  
  return attachments    

  
def get_times(times, altaz,horizon=12*u.deg):
  rise_time = times[np.where(altaz.alt>horizon)[0][0]]
  set_time = times[np.where(altaz.alt>horizon)[0][-1]]
  return rise_time, set_time


def calc_rise_time(skycoord, telescope, horizon=12*u.deg):
  delta_times = np.linspace(0,24,300)*u.hour
  time = Time.now()
  
  times = time+delta_times
  
  
  telescope_altaz = AltAz(obstime=times, location=telescope)
  
  event_altaz = skycoord.transform_to(telescope_altaz)
  rise_time, set_time = get_times(times, event_altaz)
  
  return rise_time, set_time



def calc_FAR(far_str):
  FAR = float(far_str)*31536000 #convert FAR from Hz to year^(-1)
  
  return FAR


def make_prob_txt(params):
  P_BNS = float(params['BNS'])
  P_NSBH = float(params['NSBH'])
  P_BBH = float(params['BBH'])
  P_T = float(params['Terrestrial'])
  P_MG = float(params['MassGap'])
  
  probs = np.asarray([P_BNS, P_NSBH, P_BBH, P_T, P_MG])
  prob_names = np.asarray(['BNS', 'NSBH', 'BBH', 'Terrestrial', 'MassGap'])
  
  sorted_probs = np.flip(np.argsort(probs))
  ordered_names = prob_names[sorted_probs]

  max_prob = np.max(probs)
  
  if P_BNS == max_prob:
    prob_txt = "This event is probably a *BNS merger*: "
  elif P_NSBH == max_prob:
    prob_txt = "This event is probably a *NSBH merger*: "
  elif P_BBH == max_prob:
    prob_txt = "This event is probably a *BBH merger*: "
  elif P_MG == max_prob:
    prob_txt = "This event is probably in the *Mass Gap*: "
  elif P_T == max_prob:
    prob_txt = "This event is probably *Terrestrial*:"
  
  for name,prob in zip(ordered_names, probs[sorted_probs]):
    prob_txt += "\n\tP(%s) = %.2f"%(name, prob)
    
  
  return prob_txt
  

def make_eventpage_txt(params):
  eventpage_txt = "You can find the event page here: <%s>"%(params['EventPage'])
  
  return eventpage_txt
  


def respond_preliminary(root, params):
  FAR = calc_FAR(params['FAR'])
  timing = root.findall('.//ISOTime')
  event_time = timing[0].text

  event_txt = "*Preliminary notice:* %s detected with %s and FAR %.2G /yr\n*Time of Signal:* %sUTC"%(params['GraceID'], params['Instruments'], FAR, event_time)
  
  logger.info('This is a preliminary notice of event %s'%(params['GraceID']))

  prob_txt = make_prob_txt(params)
  dist_txt, pos_txt,localisation_txt = skymap_strings(params)
  vetted_txt = "This notice *HAS NOT* been vetted by humans"
  eventpage_txt = make_eventpage_txt(params)
  
  
  
  message_txt = '%s\n\n%s\n%s\n%s\n%s\n\n%s\n\n%s'%(event_txt, prob_txt, dist_txt, localisation_txt, pos_txt, vetted_txt, eventpage_txt)
  
  return message_txt
  

def respond_initial(root, params):
  FAR = calc_FAR(params['FAR'])
  timing = root.findall('.//ISOTime')
  event_time = timing[0].text

  event_txt = "*Initial Notice:* %s detected with %s and FAR %.2G /yr\n*Time of Signal:* %sUTC"%(params['GraceID'], params['Instruments'], FAR, event_time)
  
  logger.info('This is an initial notice of event %s'%(params['GraceID']))

  prob_txt = make_prob_txt(params)
  dist_txt, pos_txt, localisation_txt = skymap_strings(params)
  vetted_txt = "This notice *HAS* been vetted by humans"
  eventpage_txt = make_eventpage_txt(params)
  
  message_txt = '%s\n\n%s\n%s\n%s\n%s\n\n%s\n\n%s'%(event_txt, prob_txt, dist_txt, localisation_txt, pos_txt, vetted_txt, eventpage_txt)
  
  return message_txt

def respond_update(root, params):
  FAR = calc_FAR(params['FAR'])

  event_txt = "*Update on %s*"%(params['GraceID'])
  logger.info('This is an update on event %s'%(params['GraceID']))

  prob_txt = make_prob_txt(params)
  dist_txt, pos_txt, localisation_txt = skymap_strings(params)
  vetted_txt = "This notice *HAS* been vetted by humans"
  eventpage_txt = make_eventpage_txt(params)
  
  message_txt = '%s\n\n%s\n%s\n%s\n%s\n\n%s\n\n%s'%(event_txt, prob_txt, dist_txt, pos_txt, localisation_txt, vetted_txt, eventpage_txt)
  
  return message_txt
  
def respond_retraction(root, params):
  event_txt = "*Retraction:* %s"%(params['GraceID'])
  logger.info('This notice is a retraction of event %s'%(params['GraceID']))

  eventpage_txt = make_eventpage_txt(params)
  
  message_txt = '%s\n\nThis event has been retracted after human vetting. %s'%(event_txt, eventpage_txt)
  
  return message_txt
  
  
def make_SMS_txt(params):
  alert_type = params['AlertType']
  sms_txt = "%s notice\n"%(alert_type)
  
  sms_txt += "%s.\n"%(make_prob_txt(params))
  sms_txt += "More info here: %s"%(params['EventPage'])
  
  return sms_txt
  
@gcn.handlers.include_notice_types(
  gcn.notice_types.LVC_PRELIMINARY,
  gcn.notice_types.LVC_INITIAL,
  gcn.notice_types.LVC_UPDATE,
  gcn.notice_types.LVC_RETRACTION
  )
def process_gcn(payload, root):
  """
  
  
  """
  
  logger.info('Alert received')
  
  event_sms = True # In general we want to send an sms for each event
  
  # Respond only to observation alerts
  if root.attrib['role'] != 'observation':
    logger.info('This event is not an observation alert')
    
    try:
      gcnbot_client.api_call("chat.postMessage", channel=gcnbot_channel, text="TEST ALERT RECEIVED")
      logger.debug('Posted test alert to test GCN slack')
    except Exception as e:
      logger.exception('Unable to post test alert to test GCN slack')
      
    if not testing:
      logger.debug('The bot is not running in testing mode. No further processing required')
      return
  
  if root.attrib['role'] == 'test':
    test_alert = True
    
    # If the alert is a test event and the code is not being run in testing mode, do not send an SMS
    if not testing:
      event_sms = False
    
    # Add disclaimer to test alerts
    for slack_client, channel in zip(client_list, channel_list): 
      slack_client.api_call("chat.postMessage", channel=channel, text="-----TEST ALERT-----")
      logger.debug('Slack: Sent test disclaimer to gcnbot test channel')
  
  else:
    test_alert = False
  
  
  # Read all of the VOEvent parameters from the "What" section.
  params = {elem.attrib['name']:
            elem.attrib['value']
            for elem in root.iterfind('.//Param')}
  
  if params['Packet_Type'] == "164":
    message_txt = respond_retraction(root, params)
    event_sms = False
    
    
  else:
    alert_type = params['AlertType']
    event_id = params['GraceID']
    
    if float(params['HasNS']) < HasNS_thresh:
      event_sms = False
      logger.info('Event %s below "HasNS" threshold'%(event_id))
    
    sms_txt = make_SMS_txt(params)
    
    if test_alert == True:
      sms_txt = "*TEST ALERT*\n%s"%(sms_txt)
    
    if alert_type == 'Preliminary':
      message_txt = respond_preliminary(root, params)
      
    elif alert_type == 'Initial':
      message_txt = respond_initial(root, params)
      
      if sms and event_sms: # Send an SMS if the general SMS flag is set AND we want to send one for this specific event
        send_SMS.respond_to_alert(sms_txt, test=testing)
        logger.debug('Sent all SMS alerts for event %s'%(event_id))
      
    elif alert_type == 'Update':
      message_txt = respond_update(root, params)
  
  message_txt += '\n\n\n'
  
  for slack_client, channel in zip(client_list, channel_list):
    slack_client.api_call("chat.postMessage", channel=channel, text=message_txt)
  logger.debug('Sent slack messages to channel list')
  
  if skymap_plot:
    skymap_plotname = 'obs_plots/%s_skymap.png'%(params['GraceID'])
    urllib.request.urlretrieve(params['skymap_png'],skymap_plotname)
    
    shutil.copyfile(skymap_plotname,'/import/www/personal/ddob1600/gcnbot/%s'%(skymap_plotname))
    
    skymap_url =  "http://www.physics.usyd.edu.au/~ddob1600/gcnbot/%s"%(skymap_plotname)
    
    attachments = [{"title": "%s skymap"%(params['GraceID']), "image_url": skymap_url}]
    for slack_client in client_list:
      slack_client.api_call("chat.postMessage", channel=channel, text='', attachments=attachments)

  if observability_plot:
    if params['Packet_Type'] != "164":
      try:
        obs_attachments = make_observability_plot(params)
        for slack_client, channel in zip(client_list, channel_list):
          slack_client.api_call("chat.postMessage", channel=channel, text='', attachments=obs_attachments)
        logger.debug('Posted observability plot in all channels')
        
      except:
        logger.debug('Error producing observability plots')
        
  logger.critical('Finished processing alert')



  
def initialise_bot(channel, tokenfile, name='Bot'):
  f = open(tokenfile,'rU')
  token = f.readlines()[0][:-1]
  client = SlackClient(token)
  
  bot_id = None
  
  if client.rtm_connect(with_team_state=False):
    logger.info("%s Connected and running!"%(name))
    # Read bot's user ID by calling Web API method `auth.test`
    bot_id = client.api_call("auth.test")["user_id"]
    
  else:
    logger.critical("%s Connection failed. Exception traceback above."%(name))
    
  return client, bot_id, channel
    
def run_sms_test():
  global observability_plot
  global skymap_plot
  global sms
  global testing
  global HasNS_thresh
  
  global client_list
  global channel_list
  
  observability_plot = True
  skymap_plot = False
  sms = True
  testing = True
  HasNS_thresh = 0.0
  
  global gcnbot_client
  global gcnbot_channel
  global gcnbot_id
  gcnbot_client, gcnbot_id, gcnbot_channel = initialise_bot('gcnbot-test', 'token', name='testbot')
  
  client_list = [gcnbot_client]
  channel_list = [gcnbot_channel]
  
  url = 'example_initial.xml'
  payload = astropy.utils.data.get_file_contents(url)
  root = lxml.etree.fromstring(payload)
  process_gcn(payload, root)
  
  
def run_output_test():
  global observability_plot
  global skymap_plot
  global sms
  global testing
  global HasNS_thresh
  
  global client_list
  global channel_list
  
  observability_plot = True
  skymap_plot = False
  sms = False
  testing = True
  HasNS_thresh = 0.0
  
  global gcnbot_client
  global gcnbot_channel
  global gcnbot_id
  gcnbot_client, gcnbot_id, gcnbot_channel = initialise_bot('gcnbot-test', 'token', name='testbot')
  
  client_list = [gcnbot_client]
  channel_list = [gcnbot_channel]
  
  url = 'example_preliminary.xml'
  payload = astropy.utils.data.get_file_contents(url)
  root = lxml.etree.fromstring(payload)
  process_gcn(payload, root)
  
  url = 'example_initial.xml'
  payload = astropy.utils.data.get_file_contents(url)
  root = lxml.etree.fromstring(payload)
  process_gcn(payload, root)
  
  url = 'example_update.xml'
  payload = astropy.utils.data.get_file_contents(url)
  root = lxml.etree.fromstring(payload)
  process_gcn(payload, root)
  
  url = 'example_retraction.xml'
  payload = astropy.utils.data.get_file_contents(url)
  root = lxml.etree.fromstring(payload)
  process_gcn(payload, root)
  
  url = 'example_bad_url.xml'
  payload = astropy.utils.data.get_file_contents(url)
  root = lxml.etree.fromstring(payload)
  process_gcn(payload, root)
  
def run_full_test():
  logger.info('Started end-to-end testing')
  
  global observability_plot
  global skymap_plot
  global sms
  global testing
  global HasNS_thresh
  
  global client_list
  global channel_list
  
  observability_plot = False
  skymap_plot = False
  sms = False
  testing = True
  HasNS_thresh = 0.0
  
  global gcnbot_client
  global gcnbot_channel
  global gcnbot_id
  gcnbot_client, gcnbot_id, gcnbot_channel = initialise_bot('gcnbot-test', 'token', name='testbot')
  
  client_list = [gcnbot_client]
  channel_list = [gcnbot_channel]
  
  logger.info('Listening for alerts...')
  
  gcn.listen(handler=process_gcn)

def fake_trigger(url = 'example_initial.xml'):
  logger.info('gcnbot deployed')
  
  global observability_plot
  global skymap_plot
  global sms
  global testing
  global HasNS_thresh
  
  global client_list
  global channel_list
  
  observability_plot = True
  skymap_plot = False
  sms = False
  testing = True
  HasNS_thresh = 0.0
  
  global gcnbot_client
  global gcnbot_channel
  global gcnbot_id
  
  #sydgw_client, sydgw_id, sydgw_channel = initialise_bot('gcnbot', 'sydgw-token', name='sydgw-bot')
  growth_client, growth_id, growth_channel = initialise_bot('alerts', 'growth-token', name='gcnbot')
  #jagwar_client, jagwar_id, jagwar_channel = initialise_bot('alerts', 'jagwar-token', name='jagwar-bot')
  gcnbot_client, gcnbot_id, gcnbot_channel = initialise_bot('gcnbot-test', 'token', name='testbot')
  
  client_list = [gcnbot_client,growth_client] 
  channel_list = [gcnbot_channel,growth_channel]

  payload = astropy.utils.data.get_file_contents(url)
  root = lxml.etree.fromstring(payload)
  process_gcn(payload, root)
  
def run_real():
  logger.info('gcnbot deployed')
  
  global observability_plot
  global skymap_plot
  global sms
  global testing
  global HasNS_thresh
  
  global client_list
  global channel_list
  
  observability_plot = True
  skymap_plot = False
  sms = True
  testing = False
  HasNS_thresh = 0.0
  
  global gcnbot_client
  global gcnbot_channel
  global gcnbot_id
  
  sydgw_client, sydgw_id, sydgw_channel = initialise_bot('gcnbot', 'sydgw-token', name='sydgw-bot')
  jagwar_client, jagwar_id, jagwar_channel = initialise_bot('alerts', 'jagwar-token', name='jagwar-bot')
  growth_client, growth_id, growth_channel = initialise_bot('alerts', 'growth-token', name='gcnbot')
  gcnbot_client, gcnbot_id, gcnbot_channel = initialise_bot('gcnbot-test', 'token', name='testbot')
  
  client_list = [growth_client, jagwar_client, sydgw_client, gcnbot_client]
  channel_list = [growth_channel, jagwar_channel, sydgw_channel, gcnbot_channel]
  
  logger.info('Listening for alerts...')
  
  gcn.listen(handler=process_gcn)

if __name__ == '__main__':
  run_real()
