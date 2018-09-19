from __future__ import print_function

class Session(object):
    """Class for setting up a session, connected to a remove server via rest APIs.
    
    :Example:
    
    >>> from westpy import *
    >>> session = Session("your.email@domain.edu")
    
    """
    def __init__(self,emailId) :
        self.token = None
        self.emailId = str(emailId)
        #
        # --- CONFIGURABLE PARAMETERS --- 
        self.serverName = "imedevel.uchicago.edu"
        self.restAPIinit = "http://imedevel.uchicago.edu:8000/init"
        self.restAPIrun  = "http://imedevel.uchicago.edu:8000/run"
        self.restAPIstop = "http://imedevel.uchicago.edu:8000/stop"
        self.restAPIstatus = "http://imedevel.uchicago.edu:8000/status"
        self.maxSessionTime = 3600 # seconds
        self.maxWaitTime = 1800 # seconds
        self.maxNumberOfCores = 4
        self.allowedExecutables = ["pw","wstat","wfreq"]
        # -------------------------------
        #
        data = {'emailId': self.emailId ,'sessionTime':str(self.maxSessionTime)} 
        #
        import requests
        import json
        #
        response = None
        try:
            output = requests.post(self.restAPIinit, data=json.dumps(data))
            response = json.loads(output.text)
        except Exception as e:
            print('The server is not responding.',e)
        if response:
            if "Error" in response:
                print("Server failed with the following error \n",response['Error'])
                return None
            else:
                print("Check the inbox/spam folder of your email and click on the link to activate the session")
                self.token = response["token"]
        else:
            print('The server is not responding.')
    
    def getToken(self):
        """Returns the token of the session.
        
        :Example:
        
        >>> from westpy import *
        >>> session = Session("your.email@domain.edu")
        >>> token = session.getToken()
        >>> print(token)
        
        """
        if self.token :
            return self.token
        else:
            raise ValueError("Cannot find output.")
    
    def stop(self):
        """Stops the session and clears the remote workspace. 
        
        :Example:
        
        >>> from westpy import *
        >>> session = Session("your.email@domain.edu")
        >>> session.stop()
        
        """
        
        import requests
        import json
        #
        headers = {'Content-Type':'application/json; charset=utf-8','emailId':self.emailId,'token':self.token}
        try:
            response = requests.get(self.restAPIstop, headers=headers, timeout=None)
        except Exception as e:
            print('The server is not responding.',e)		 
        return json.loads(response.text) 

    def status(self):
        """Returns whether the session is active and time left. 
        
        :Example:
        
        >>> from westpy import *
        >>> session = Session("your.email@domain.edu")
        >>> session.status()
        
        """
        
        import requests
        import json
        #
        headers = {'Content-Type':'application/json; charset=utf-8','emailId':self.emailId,'token':self.token}
        try:
            response = requests.get(self.restAPIstatus, headers=headers, timeout=None)
        except Exception as e:
            print('The server is not responding.',e)		 
        return json.loads(response.text) 
		
    def run(self,executable=None,inputFile=None,outputFile=None,downloadUrl=[],number_of_cores=2) :
        """Runs the executable on the remote server.
        
        :param executable: name of executable
        :type executable: string
        :param inputFile: name of input file
        :type inputFile: string
        :param outputFile: name of output file
        :type outputFile: string
        :param downloadUrl: URLs to be downloaded
        :type downloadUrl: list of string
        :param number_of_cores: number of cores
        :type number_of_cores: int
        
        :Example:
        
        >>> from westpy import *
        >>> session = Session("your.email@domain.edu")
        >>> session.run( "pw", "pw.in", "pw.out", ["http://www.quantum-simulation.org/potentials/sg15_oncv/upf/C_ONCV_PBE-1.0.upf"] , 2 )
        >>> session.stop()
        
        """
        #
        import json
        #
        assert( number_of_cores <= self.maxNumberOfCores )
        #
        output_dict = {}
        if executable in self.allowedExecutables : 
           # set inputs
           if inputFile is None:
              inputFile = str(executable)+".in"
           if outputFile is None:
              outputFile = str(executable)+".out"		 
           try:
              output = self.__runExecutable(executable,inputFile,downloadUrl,number_of_cores)
              output_json = json.loads(output)
              if "Error" in output_json:
                 print("Server failed with the following error \n",output_json['Error'])
                 return None			   
              elif "JOB DONE." not in str(output_json['output']).strip():
                 print("MPI execution failed with the following error:  \n"+str(output))
                 return None
              output_data = str(output_json['output']).strip()
              if "pw" in executable:
                 output_dict = json.loads(output_json['output_dict'])
              else:
                 output_dict = output_json['output_dict']
              # write the output file
              with open(outputFile, "w") as file :
                 file.write(str(output_data))
           except Exception as e:
              print("Session Expired! Invalid Request sent, Please recreate session and recheck your input. \n"+ str(e))		 
              return None          
        else:
           raise ValueError("Invalid executable name") 
        #
        print("Generated ",outputFile)
        return output_dict
      
    def __runExecutable(self,executable,input_file,download_urls,number_of_cores) :
        """Runs remotely the executable using a REST api.
        """
        #
        import requests
        import json
        # suck in the input file
        try:
           file_content = ""	     
           with open(input_file,'r') as f :
              for line in f :
                 file_content = file_content + line + "\\n"
        except FileNotFoundError:
           error = "Could not find "+ input_file + ". \n Generate input file "+ input_file +" and try again."	  
           print(error)
           return None
        body = {'urls':download_urls,'file':file_content,'cmd_timeout':str(self.maxWaitTime),'script_type':str(executable),'no_of_cores':str(number_of_cores)}  
        jsondata = json.dumps(body)
        jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
        headers = {'Content-Type':'application/json; charset=utf-8','emailId':self.emailId,'token':self.token}
        response = ""
        print("Requested to run executable", executable, "on server", self.serverName)
        try:
           response = requests.post(self.restAPIrun, data = jsondataasbytes, headers=headers, timeout=None)
           return response.text
        except Exception as e:
           print('The server is not responding.',e)	
           return None
        return None
