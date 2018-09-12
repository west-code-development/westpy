import json
import requests
class Connection(object):
	"""Class for setting up the Connection to the Rest API Server.

	:Example:

	>>> from westpy import *
	>>> session = Connection("emailId")
	"""
	def __init__(self,emailId=None) :
		self.output = {}
		response = None
		if emailId:
			self.emailId = str(emailId)
			data = {'emailId':str(emailId),'sessionTime':str(600)}
			try:
				output = requests.post("http://imedevel.uchicago.edu:8000/getSessionId", data=json.dumps(data))
				response = json.loads(output.text)
			except Exception as e:
				print('The West Rest API is not responding, Please try again later',e)
			if response:
				if "Error" in response:
					print("Execution failed with the following error \n",response['Error'])
					return None
				else:
					print("Check the inbox/spam folder of your email and click on link to activate session")
					self.output = response
			else:
				print('The West Rest API response to longer than usual, Please try again later')
		else:
			print("Email Id is mandatory to establish Connection")
	
	
	def getOutput(self):
		"""returns the token to setup the session.
		"""
		if self.output:
			return self.output
		else:
			raise ValueError("No Email id found")

	def stop(self):
		"""stops the executable using the token

		:Example:

		>>> session = Session("emailId")	
		>>> gs.run( session, "pw" )
		>>> session.stop()

		"""

		headers = {'Content-Type':'application/json; charset=utf-8','emailId':self.output['emailId'],'token':self.output['token']}
		try:
			response = requests.get("http://imedevel.uchicago.edu:8000/stopSession", headers=headers, timeout=None)
		except Exception as e:
			print('The West Rest API response to longer than usual, Please try again later',e)		 
		return json.loads(response.text)      

		