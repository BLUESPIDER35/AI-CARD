!apt-get update
!apt-get install -y chromium-browser
!apt install chromium-chromedriver
!ls /usr/lib/chromium-browser/
!pip install -r /content/drive/MyDrive/baseball/requirements.txt

from os import path
from selenium import webdriver
import undetected_chromedriver as uc

options = uc.ChromeOptions()
options.add_argument('--incognito')
options.add_argument("--window-size=1920,1080")
options.add_argument("--start-maximized")

options.add_argument("--disable-extensions")
options.add_argument("--disable-application-cache")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-setuid-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument('--headless=new')

driver = webdriver.Chrome(options=options)
driver.get('https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python')
print(driver.title)


"""handle backend tasks with redis database"""

import redis
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pydantic import BaseModel
from typing import Any, Union, Dict, List
from redis.exceptions import ResponseError
from abc import ABCMeta, abstractmethod
from datetime import datetime
import numpy as np
from pprint import pprint


class DBSettings(BaseModel):
    """create the settings for the base model for the user database"""
    expiration_period: int = 172800
    username: str = 'doadmin'
    password: str = '7X52yc0N8Z94bvY1'
    host: str = f"mongodb+srv://{username}:{password}@db-mongodb-nyc1-60998-0cb266db.mongo.ondigitalocean.com"
    db_url: str = None
    sport_db_idx: dict = {key: value for key, value in zip(['Football', ' Basketball', 'Tennis',
                                                         'AmericanFootball', 'Baseball', 'Hockey'], range(1, 7))}
    sport_db_idx['users'] = 0 # set user index
    upsert_upon_update: bool = True
    allowed_search_keys: list = ['search_id', 'Sport', 'match_id']


class SingletonType(type):
    """prevent duplicate instantiations from being created in the DB class by singleton design pattern"""
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# abstract factory method for filtering the query results manually
class FilterQuery(metaclass=ABCMeta):
    """metaclass for filtering the query result from the database key search"""

    def __init__(self):
        pass

    @abstractmethod
    def filter(self, query_results: list, **kwargs) -> Union[dict, list]:
        """filter the results from the database search based on strategy"""
        pass


class NoFilter(FilterQuery):
    """apply no filter and return full results"""

    # apply the filtering  in particular method
    def filter(self, query_results: list, **kwargs) -> Union[dict, list]:
        # return full results without any filtering
        return query_results


class SportsbookFilter(FilterQuery):
    """filter based on an array of specific sportsbooks"""

    def filter(self, query_results: list, **kwargs) -> Union[dict, list]:
        """filter based on a specific set of sportsbooks"""

        # make a copy of query results
        new_query_results = query_results.copy()

        # iterate through each match
        flag: bool = False
        for idx, match in enumerate(query_results):
            for key, packets in match['odds'].items(): # specific odd types and enter odds data branch
                # define the filtered container
                container: list = []
                for idx_p, packet in enumerate(packets): # get the specific odds by sportsbook in that odd type package
                    # check whether odd type has lines involved
                    flag = False
                    if 'line' in packet.keys():
                        flag = True
                        # iterate through the internal line odds
                        moneyline_container: list = []
                        for line_value in packet['values']: # line value is of dictionary type
                            if line_value['odds']['sportsbook'] in kwargs['filter_value']: # check for sportsbook filter
                                moneyline_container.append(line_value)

                        new_query_results[idx]['odds'][key][idx_p]['values'] = moneyline_container
                    else:
                        if packet['odds']['sportsbook'] in kwargs['filter_value']:
                            container.append(packet)

                # overwrite the existing query results with the filtered results
                if not flag:
                    new_query_results[idx]['odds'][key] = container

        return new_query_results


class TournamentFilter(FilterQuery):
    """filtering queries by tournament"""

    def filter(self, query_results: list, **kwargs) -> Union[dict, list]:
        """filter for a specific kind of tournament specified in the query"""

        # define the output container
        container: list = []

        # iterate through matches and then filter by tournament
        for match in query_results:
            # check if the tournament of the query result matches with the requested tournament filter
            if match['Tournament'].lower().strip() in [i.lower().strip() for i in kwargs['filter_value']]:
                # add match to the container of results
                container.append(match)

        return container


class DateFilter(FilterQuery):
    """filter for all matches after or before a certain date"""

    def filter(self, query_results: list, **kwargs) -> Union[dict, list]:
        """filter all results based on whether they occur before or after a specified date"""

        # define the output container
        container: list = []

        # gather requested filters
        reference_date, before = kwargs['filter_value']

        # convert the reference date to a datetime object and check for formatting error
        try:
            reference_date = datetime.strptime(reference_date, "%m-%d-%Y")
        except ValueError:
            return query_results

        # iterate through matches and check which dates follow condition
        for match in query_results:
             # get the match date
            match_date = datetime.strptime(match['Date'], "%m-%d-%Y")

            # conditionals to filter for match based on dates
            if before:
                if match_date < reference_date:
                    container.append(match)
            else:
                if match_date >= reference_date:
                    container.append(match)
        return container


class TeamFilter(FilterQuery):
    """filter for a specific team playing"""

    def filter(self, query_results: list, **kwargs) -> Union[dict, list]:
        """filter for a team and check both opponents for full functionality"""

        # define output container
        container: list = []

        # iterate through matches and check teams involved
        for match in query_results:
            # check how many teams are being filtered for
            if len(kwargs['filter_value']) == 1:
                # check whether the filtered team is found in the match teams
                if kwargs['filter_value'][0] in [match['TeamA'], match['TeamB']]:
                    # add match to the output container
                    container.append(match)
            elif len(kwargs['filter_value']) == 2:
                # check whether the team array is equal
                if (np.array(kwargs['filter_value']) == np.array([match['TeamA'], match['TeamB']])).all() or (
                        np.array(kwargs['filter_value']) == np.array([match['TeamB'], match['TeamA']])).all():
                    # add match to the output container
                    container.append(match)
            else:
                # if more than 2 conditions are supplied then do not apply a filter
                container.append(match)

        return container


class OddTypeFilter(FilterQuery):
    """filter for a specific type of odd (home/away, over/under ...)"""

    def filter(self, query_results: list, **kwargs) -> Union[dict, list]:
        """filter for a specific type of odd"""

        # define filtered odd types
        odd_types: list = kwargs['OddTypes']

        # iterate through matches and apply the odd type filter
        for idx, match in enumerate(query_results):
            # define a temporary container for the odds branch
            temp_container: dict = {}

            # iterate through the available odds for the match
            for key, value in match['odds'].items():
                # check if the odd type condition is satisfied
                if key in odd_types:
                    temp_container[key] = value

            # rewrite the query results object to include the new conditions
            query_results[idx]['odds'] = temp_container

        return query_results


# abstract factory method for creating query for deleting document
class DelQuery(metaclass=ABCMeta):
    """abstract metaclass for deletion queries based on use case"""
    def del_query(self, identifiers: dict) -> dict:
        pass


class UserDelQuery(DelQuery):
    """deletion query for the users database"""

    def del_query(self, identifiers: dict) -> dict:
        """deletion query based on the user key / api key"""
        try:
            # construct mongodb query based on singular condition
            return {'user_key': identifiers['user_key']}
        except KeyError:
            # raise error if incomplete data packet is supplied
            raise Exception("404: User Key Not Found")


class SportsDelQuery(DelQuery):
    """deletion query for the sports database"""

    def del_query(self, identifiers: dict) -> dict:
        """deletion query based on the sport and the time of the event"""
        try:
            # construct the mongodb with $AND operator
            return {"$and": [{"Sport": identifiers['sport']},
                          {"Date": identifiers['date']}]}
        except (IndexError, KeyError):
            # raise error if incomplete data packet is supplied
            raise Exception("404: Sports Key Not Found")


class ConnectDB(metaclass=SingletonType):
    """main class for initiating database access and connection"""

    def __init__(self, use_case: str, **kwargs) -> None:
        # establish settings collection
        self.settings = DBSettings()
        self.use_case = use_case

        # create database connection to redis if url is available otherwise use localhost
        try:
            self.client = MongoClient(self.settings.host)
            self.db = self.client[self.use_case]
            self.collection = self.db[kwargs['collection']]
        except (ConnectionFailure, KeyError):
            raise("Connection Failed")

        # gather the mapping to use for abstract factory method calling
        self.filter_maps, self.del_maps = self._create_abstract_factory_mappings()

    @staticmethod
    def _create_abstract_factory_mappings() -> List[dict]:
        """create mapping for the abstract factory methods: filtering, deletion queries"""

        # create mapping for the query filters
        filter_q_idx: Dict[int, Any] = {
            1: NoFilter(),
            2: SportsbookFilter(),
            3: TournamentFilter(),
            4: DateFilter(),
            5: TeamFilter(),
            6: OddTypeFilter()
        }

        # create mapping for the deletion filters
        del_query_idx: Dict[int: Any] = {
            'Users': UserDelQuery(),
            'Sports': SportsDelQuery()
        }

        return [filter_q_idx, del_query_idx]

    def _check_connection(self) -> Any:
        """check whether the connection to the database is established"""
        return self.client.server_info()

    def __add__(self, package: dict) -> bool:
        """add an entry to the database by magic method"""

        # add data to redis
        try:
            if len(list(self.collection.find({'user_key': package['user_key']}))) == 0:
                self.collection.insert_one(package)
            # output success code
            return True
        except (MemoryError, ValueError, KeyError): # handle for lack of memory or value errors
            # output failure code
            return False

    def __sub__(self, identifiers: dict) -> bool:
        """remove an entry from the database by magic method"""

        # define the query to remove a document from the collection
        del_query: dict = self.del_maps[self.use_case].del_query(identifiers)

        try:
            # delete the document from the collection
            self.collection.delete_many(del_query)
            # output success code
            return True
        except (ValueError, KeyError): # handle for potentially not found keys in database
            # output failure code
            return False

    def __setitem__(self, user_key: str, user_calls: int) -> bool:
        """edit a user's personal authentication data"""

        # define search query for the document
        update_query: dict = {'user_key': user_key}

        try:
            # update the item in the collection with specific targeting to one document
            self.collection.update_one(update_query, {'$set': {'user_calls': user_calls}},
                                       upsert=self.settings.upsert_upon_update) # allow for upsert to avoid errors
            # return success code
            return True
        except (MemoryError, ValueError, KeyError):
            # return failure code
            return False

    def _filter_query(self, query_results: list, filtering_options: list, filtering_values: list):
        """filter the results of the query based on specified strategies"""

        # iterate through the filtering objects and apply to the queried results
        for filter_strategy, filter_value in zip(filtering_options, filtering_values):
            # run abstract factory methods
            query_results = self.filter_maps[filter_strategy].filter(query_results, filter_value=filter_value)

        return query_results

    def retrieve_api_calls(self, user_key: str) -> int:
        """retrieve the status of a user"""

        # create the search query
        search_query: dict = {'user_key': user_key}

        # run the query
        try:
            # search for specific document based on the user key
            return int(list(self.collection.find(search_query))[0]['user_calls'])
        except (TypeError, ResponseError, IndexError, KeyError):
            # return 0 when failed since document does not exist
            return 0

    def get_all_items(self) -> list:
        """get all keys from database"""
        return [str(i) for i in list(self.collection.find())]

    def retrieve_sports_data(self, key: str, value: str, filters: List[int] = [1], filter_values: list = [None]) -> Any:
        """execute a redis query to retrieve the requested data"""

        # check whether the key is acceptable
        if key not in self.settings.allowed_search_keys:
            raise ("403: Invalid Use Case Supplied")

        # define the search query for a particular match or set of matches
        search_query: dict = {key: value} # where key can be either or match_id when checking for uniqueness

        # query the database for all search terms related to the sport id
        query_data: Any = list(self.collection.find(search_query))

        # apply filtering to the query
        return self._filter_query(query_data, filters, filter_values)


class ScraperDB(ConnectDB):
    """database class for the scraped results"""

    def __init__(self, use_case: str, **kwargs):
        super().__init__(use_case, **kwargs)
        self.use_case = use_case

    def _fetch_non_missing(self, match: dict):
        """get the non-missing odds headers to set in the database"""

        # define the output container
        container: list = []

        # try to get the odds values
        try:
            match_odds: dict = match['Odds']
        except KeyError:
            raise BrokenPipeError # raise a pipeline error if there is a match with no odds

        # iterate through each key in the match odds and check if there are odds available
        for key, value in match_odds.items():
            # check if there are odds value to set in database
            if len(value) != 0:
                container.append(key) # add the key to the container if found valid

        return container

    def _create_match_id(self) -> int:
        """create a match id for a specific instance"""

        # get the number of items in the collection
        col_count: int = self.collection.count_documents({})

        return col_count + 1

    def upload_results(self, matches: list) -> None:
        """add data on a specific sports match"""

        # iterate through all the matches to insert each one depending on it's uniqueness
        for data_blob in matches:
            # create the match id based on the data provided in the packet
            search_id = f'{data_blob["Sport"]} {data_blob["OpponentA"]}-{data_blob["OpponentB"]}-'\
                            f'{data_blob["Tournament"]}-{data_blob["Date"]}'

            # error handling for the match id
            try:
                data_blob['search_id'] = search_id  # add the match id to the database
            except KeyError:
                print("ERROR: ")
                pprint("Corresponding blob: ", data_blob)
                continue

            # add the timestamp to the document
            data_blob['timestamp'] = str(datetime.now())

            # create the match id
            match_id = self._create_match_id()
            data_blob['match_id'] = match_id

            # check for uniqueness
            if len(self.retrieve_sports_data('search_id', search_id)) == 0:
                # then update the collection with the item
                self.collection.insert_one(data_blob)
            else:# update the existing document in the collection
                # run code to find valid keys to update in the collection
                try:
                    valid_keys: list = self._fetch_non_missing(data_blob)
                except KeyError:
                    print(data_blob)
                    valid_keys: list = []

                # iterate through the non-empty keys
                for key in valid_keys:
                    self.collection.update_one({'search_id': search_id}, {'$set': {
                        f'Odds.{key}': data_blob['Odds'][key], 'Date': data_blob['Date']}})
                    




"""handle API key creation and user authentication"""
from uuid import uuid4
from abc import ABCMeta, abstractmethod
from pydantic import BaseModel
from typing import Union

class UserCodes(BaseModel):
    """user codes for each payment plan to conserve DB space in beginning stages of development"""
    tier_codes: dict = {1: 'a13Px',
                        2: 'gh52o',
                        3: 'pnl4u',
                        4: 'y4v7i'}
    tier_calls: dict = {1: 25,
                        2: 50,
                        3: 150,
                        4: 250}
    regenerate_period_days: int = 1 # how often api calls are restored to the account


class Authentication(metaclass=ABCMeta):
    """authenticate and manage a specific user"""

    def __init__(self):
        self.user_codes = UserCodes()

    @abstractmethod
    def user_metrics(self, user_key: str, **addition_user_data) -> bool:
        pass


class RegenerateCalls(Authentication):
    """regenerate api calls after one day for user"""

    def _compute_payment_plan(self, user_key: str) -> int:
        """check to which payment plan a user is subscribed"""

        # fetch the last 5 digits of the api token
        user_code: str = user_key[-5:]

        # search for user tier and map to available calls for user
        user_calls: int = [self.user_codes.tier_calls[key] for key, item in
                 self.user_codes.tier_codes.items() if item == user_code][0]
        return user_calls

    # noinspection PyTypeChecker
    def user_metrics(self, user_key: str, **addition_user_data) -> bool:
        """regenerate the api calls for user after set period"""

        # get number of available calls for user by plan
        user_calls_new: int = self._compute_payment_plan(user_key)

        # get current calls and add them to plan to prevent waste
        user_calls_current = ConnectDB('Users', collection='UserData').retrieve_api_calls(user_key)

        # create user data parcel
        ConnectDB[user_key] = user_calls_current + user_calls_new


class ApproveCalls(Authentication):
    """approve an api call from a user"""

    def user_metrics(self, user_key: str, **addition_user_data) -> bool:
        """check if api credits cover the api call by a user"""

        # retrieve api calls from backend
        user_calls: int = ConnectDB('Users', collection='UserData').retrieve_api_calls(user_key)

        # check api call logic
        if user_calls - 1 >= 0:
            return True
        else:
            return False


class UpdateCalls(Authentication):
    """update a users api calls upon request"""

    # noinspection PyTypeChecker
    def user_metrics(self, user_key: str, **addition_user_data) -> bool:
        """update the api calls of the user"""

        # retrieve user data and charge api call cost
        user_calls: int = ConnectDB('Users', collection='UserData').retrieve_api_calls(user_key)
        user_calls = addition_user_data['call_cost'] - 1 if user_calls > 0 else 0

        # create new user data parcel and update in database
        ConnectDB[user_key] = user_calls


class SubscribeUser(Authentication):
    """subscribe a new user"""

    def _create_api_key(self, payment_tier: Union[int, None]) -> str:
        """create a token for the api user"""

        # create individual building blocks of token sequence
        unique_key_whole: str = str(uuid4())

        # create the full token and output
        return f'{unique_key_whole}{self.user_codes.tier_codes[payment_tier]}'

    def user_metrics(self, user_key: Union[str, None], **addition_user_data) -> bool:
        """register user in database"""

        # create an api access token for user if necessary
        user_key: str = self._create_api_key(addition_user_data['payment_tier']) if not user_key else user_key

        # add the user to the database
        return ConnectDB('Users', collection='UserData') + {'user_key': user_key, 'calls':
            self.user_codes.tier_calls[addition_user_data['payment_tier']]}


class UnSubscribeUser(Authentication):
    """unsubscribe a user from the api database"""

    def user_metrics(self, user_key: str, **addition_user_data) -> bool:
        """remove a user from the database"""

        # retrieve deletion method from backend
        return ConnectDB('Users', collection='UserData') - {'user_key': user_key}
    






    """handle conversion of different types of odds based on user input"""

from state_machine import State, Event, acts_as_state_machine, after, InvalidStateTransition
from pydantic import BaseModel
from typing import Callable, Union, Any


# create settings model for the state machine settings
class StateSettings(BaseModel):
    """settings for the state machine stored"""
    supplied_format: State = State(initial=True)
    american_odds: State = State()
    european_odds: State = State()
    fractional_odds: State = State()
    implied_probabilities: State = State()

    class Config:
        """config class for arbitrary types"""
        arbitrary_types_allowed = True

@acts_as_state_machine
class OddsConverter:

    # define the state settings object
    state_settings = StateSettings()

    supplied_format: State = State(initial=True)
    american_odds: State = State()
    european_odds: State = State()
    fractional_odds: State = State()
    implied_probabilities: State = State()

    # create the events
    to_american: Event = Event(from_states=supplied_format, to_state=american_odds)
    to_implied_probabilities: Event = Event(from_states=supplied_format,
                                            to_state=implied_probabilities)
    to_european: Event = Event(from_states=supplied_format, to_state=european_odds)
    to_fractional_odds: Event = Event(from_states=supplied_format,
                                      to_state=fractional_odds)

    def __init__(self, odds: Union[str, float, int]):
        """supplied format is always decimal odds from backend api unless different data provider"""
        self.odds: Union[int, float, str] = odds

    @staticmethod
    def call_event(event: Callable) -> None:
        """call the state machine event"""
        event()

    @after('to_american')
    def convert_american(self) -> None:
        """convert supplied odds to american odds"""

        # check decimal odds conditions satisfied and compute american odds
        if self.odds >= 2:
            self.odds = (self.odds - 1) * 100
        else:
            self.odds = (-100) * (self.odds - 1)

    @after('to_european')
    def convert_european(self) -> None:
        """convert to european odds - just return"""
        self.odds = self.odds

    @after('to_implied_probabilities')
    def convert_implied_probabilities(self) -> None:
        """convert to implied probabilities given decimal odds"""
        self.odds = 1 / self.odds

    @after('to_fractional')
    def convert_fractional(self) -> None:
        """convert to fractional odds"""
        self.odds = f'{(self.odds - 1)}/1'


def transition(process, event, event_name):
    """handle transitions between states"""
    try:
        event()
    except  InvalidStateTransition:
        print(f'Error: transition of {process.name} from {process.current_state} to {event_name} failed')

# noinspection PyUnresolvedReferences
def unit_states_controller(conversion_opt: Any,
                           conversion_class: Callable) -> Union[str, int, float]:
    """controller function for the odds conversion algorithm"""

    # run the state machine and retrieve the processed odds
    try:
        # call the event internally
        conversion_class.call_event(conversion_opt)
        return conversion_class.odds
    except (InvalidStateTransition, ValueError, TypeError):
        raise BufferError # raise a buffer error when state machine event is found
    




    """manage the parsing functionalities for the backend scraper"""

from abc import ABCMeta, abstractmethod
from pydantic import BaseModel
from typing import List, Any, Iterable, Union
import openai
from queue import PriorityQueue
from time import perf_counter
import numpy as np


class ParserSettings(BaseModel):
    """parser settings collection"""

    # parameters for over/under items and their weights
    item_count_ou: int = 10
    item_weight_ou: int = item_count_ou

    # parameters for handicap items and their weights
    item_count_hc: int = 7
    item_weight_hc: int = item_count_hc

    # grouping configurations for ruleset approach
    odds_type_index: dict = {
        'home/away': {
            'count': 2, # home and away index
            'keys': ['home', 'away'],
            'low_keyword': 'bookmakers',
            'high_keyword': 'average'
        },
        'over/under': {
            'count': 2, # over under index
            'keys': ['over', 'under'],
            'low_keyword': 'payout',
            'high_keyword': 'inactive'
        }
    }


class Parser(metaclass=ABCMeta):
    """metaclass for the parser abstract factory method"""

    def __init__(self):
        """initialize the main objects used in the factory method"""
        self.settings = ParserSettings()

    def shorten_data_blob(self, data_blob: Union[Iterable, List], odds_type) -> List[str]:
        """shorten the data blob to minimize runtime of gpt model"""

        # create variables to track beginning and ending indices
        lower, upper = None, None

        # create new container
        container: list = []
        low_keyword, high_keyword = self.settings.odds_type_index[odds_type]['low_keyword'], \
                                    self.settings.odds_type_index[odds_type]['high_keyword']

        data_blob = [i.text for i in data_blob]

        # iterate through each line in the data blob
        for idx, line in enumerate(data_blob):
            # identify keywords to cut data blob into smaller portions
            if low_keyword in line.lower():
                lower = idx
            elif high_keyword in line.lower():
                upper = idx
                break

        for idx in range(lower, upper):
            container.append(data_blob[idx])

        # cut data blob at computed indices
        return container

    def moneyline_parse(self, books: Union[Iterable, List[str]], odds: Iterable, **kwargs):
        """process the books and odds for each of those books in over/under mode"""

        # find how many percentages there are in the list
        arr: Iterable = np.array(odds)

        # get number of results in list that contain the % symbol and shorten the odds array and handle errors
        try:
            res: int = len(np.where(np.char.find(np.char.lower(arr), '%') > -1)[0])
        except TypeError:
            res: int = 0
        odds = odds[:-3 * res] if res != 0 else odds # make sure res is not zero before filtering to avoid null set
        odds_grouped = zip(*(iter(odds),) * kwargs['odds_type_n'])

        # organize odds by sports book
        return {book: [i for i in odd] for book, odd in zip(books, odds_grouped)}

    @abstractmethod
    def parse(self, data_blob: Union[Iterable, List], **kwargs) -> Union[list, dict]:
        """parse through data blob"""
        pass


class GPTParser(Parser):
    """parser method for the GPT model"""

    @staticmethod
    def _create_blob_str(data_blob) -> str:
        """create the data packet for gpt to interpret"""
        data_str: str = ''
        for line in data_blob:
            # remove new lines to minimize size of data input
            if line != '\n':
                data_str += line.replace('\n', ' ')
        return data_str

    @staticmethod  # refactor as abstract factory method
    def _parse_gpt_response(gpt_response: str, **kwargs) -> dict:
        """parse through the chatgpt response from data blob computation"""

        # create container for results
        gpt_container: dict = dict()

        # iterate through each line
        for gpt_line in gpt_response.split('\n'):
            bookmaker: str = gpt_line.split(':')[0]

            if kwargs['odds_type'] == 'home/away':
                try:
                    home: float = float(gpt_line.split(':')[1].split(',')[0])
                except TypeError:
                    continue

                try:
                    away: float = float(gpt_line.split(':')[1].split(',')[1])
                except TypeError:
                    continue

                # add data to odds container
                gpt_container[bookmaker] = {'home': home, 'away': away}

        return gpt_container

    def parse(self, data_blob: Union[Iterable, List], **kwargs) -> dict:
        """parser for the gpt model to process data blob"""

        # enter api portal with key
        openai.api_key = kwargs['gpt_api_key']

        # shorten the data blob to make gpt computation faster
        shortened_blob: List[str] = self.shorten_data_blob(data_blob, kwargs['odds_type'])

        # remove new line strings and replace them in shortened data blob
        gpt_packet: str = self._create_blob_str(shortened_blob)

        #  open conversation with gpt model and retrieve the results
        completion = openai.ChatCompletion.create(
            model=kwargs['gpt_model_id'],
            temperature=kwargs['gpt_temperature'],
            messages=[
                {"role": "user",
                 "content": f"""{kwargs['gpt_command']}{gpt_packet}"""}
            ]
        )[0].message.content

        # shorten GPT response into dictionary depending on types of odds desired
        return self._parse_gpt_response(completion, odds_type=kwargs['odds_type'])


class RuleParserHA(Parser):
    """parsing character blob with specific set of rules"""

    def _create_output_packet(self, data_container: List[Any], books: Iterable, odds_type: str) -> dict:
        """create the output packet depending on the """

        # group container into odds buckets
        buckets: int = self.settings.odds_type_index[odds_type]['count']
        data_grouped = zip(*(iter(data_container),) * (buckets))

        # create new container to store final results
        container: dict = dict()

        try:
            for book, group in zip(books, data_grouped):
                # create the data packet automatically using keys from the settings
                container[book] = {self.settings.odds_type_index[odds_type]['keys'][i]: group[i]
                                       for i in range(1, buckets)}
        except (KeyError, IndexError):
            raise ArithmeticError
        return container

    def parse(self, odds: Union[Iterable, List], **kwargs) -> dict:
        """parser function for ruleset"""

        # get the optional books argument
        books: list = kwargs['books']
        return self.moneyline_parse(books, odds, odds_type_n=kwargs['odds_group_n'])


class RuleParserMNL(Parser):
    """rule parser for over under data blobs"""

    @staticmethod
    def _handle_prior_queue_model(q: PriorityQueue, item_count: int = 10, item_weight: int = 10):
        """handle the priority queue and the document model produced"""

        # create container to store results
        container: list = list()

        # loop through each item in the priority queue
        for count, _ in enumerate(range(len(q.queue))):
            item = q.get() # pop item from the queue
            if count+1 <= item_count:# and abs(item[0]) > item_weight: # check if enough items have been retrieved valid
                container.append(item[-1]) # store item in the container to return as a list

        return container

    def _moneyline_selection_hc(self, data_blob: Union[Iterable, List], weights: Iterable) -> PriorityQueue:
        """handicap moneyline selection"""
        # create the priority queue
        q = PriorityQueue()
        # define the weights and combine them to create raw data to read
        data: list = [f'{i} {j}' for i, j in zip(data_blob, weights)]
        # iterate through data text and organize the priority queue
        for idx, line in enumerate(data):
            if line == 'BONUS' or line == '\n' or 'point' in line.lower():
                continue
            moneyline = line.split(' ')[2].strip().replace('\n', '').replace('+', '')
            if '.' in moneyline and '.5' in moneyline:
                moneyline += '0'
            elif '.' not in moneyline:
                moneyline += '.00'
            try:
                q.put((int(line.split(' ')[-1]) * -1, moneyline))
            except (TypeError, IndexError):
                continue
        return q

    def _moneyline_selection_ou(self, data_blob: Union[Iterable, List], weights: Iterable) -> PriorityQueue:
        """technique for the moneyline approach selecting top moneylines for over under to extract"""
        # create the priority queue
        q = PriorityQueue()

        # define the weights and combine them to create raw data to read
        data: list = [f'{i} {j}' for i, j in zip(data_blob, weights)]

        # iterate through data text and organize the priority queue
        for idx, line in enumerate(data):
            if line == 'BONUS' or line == '\n' or 'point' in line.lower():
                continue
            moneyline = line.split(' ')[1].replace('+', '').strip().replace('\n', '')
            if '.' in moneyline and '.5' in moneyline:
                moneyline += '0'
            elif '.' not in moneyline:
                moneyline += '.00'
            try:
                q.put((int(line.split(' ')[-1]) * -1, moneyline))
            except (TypeError, IndexError, UnboundLocalError):
                continue
        return q

    def parse(self, data_blob: Union[Iterable, List], **kwargs) -> Union[list, dict]:
        """parser for the over under rule based approach"""
        # define output by the role the function plays in the workflow of over/under scraping
        if kwargs['role'] == 1: # role when fetching raw links to then scrape
            if kwargs['method'] == 'over/under': # over/under
                q: PriorityQueue = self._moneyline_selection_ou(data_blob, kwargs['weights'])
                # get the data from the queue
                queue_data = self._handle_prior_queue_model(q,item_count=self.settings.item_count_ou,
                                        item_weight=self.settings.item_weight_ou)
                return [[f'{kwargs["base_link"]}#over-under;{kwargs["url_key"]};{i};0' for i in queue_data],
                       [i for i in queue_data]]
            else: # handicap method
                q: PriorityQueue = self._moneyline_selection_hc(data_blob, kwargs['weights'])
                queue_data = self._handle_prior_queue_model(q,item_count=self.settings.item_count_hc,
                                                            item_weight=self.settings.item_weight_hc)
                return [[f'{kwargs["base_link"]}#ah;{kwargs["url_key"]};{i};0' for i in queue_data],
                        [i for i in queue_data]]
        elif kwargs['role'] == 2:
            return self.moneyline_parse(data_blob, kwargs['odds'], odds_type_n=2) # books, odds in that order with count
        else:
            raise Exception('Role Not Found')
        







        """OddsPortal scraper for real time data feed"""
import random
from pydantic import BaseModel
import undetected_chromedriver as uc
from selenium import webdriver
from pprint import pprint
from pathlib import Path
import tomli
from tqdm import tqdm
import numpy as np
#from modelsBaseball.scrape_parser import GPTParser, RuleParserHA, RuleParserMNL
#from dbBaseball import ScraperDB
from time import sleep
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from typing import List, Dict, Any, Iterable, Union, Tuple, Callable
import openai
from concurrent.futures import ThreadPoolExecutor
import warnings
import pandas as pd
from numpy import nan
import warnings
warnings.filterwarnings("ignore")


# create selector settings for each class
class SelectorSettings(BaseModel):
    """settings for all selector operations internally"""

    # login settings
    login: dict = {
        'username': {
            'by': By.NAME,
            'selector': 'login-username'
        },
        'password': {
            'by': By.NAME,
            'selector': 'login-password'
        },
        'submit': {
            'by': By.XPATH,
            'selector': '//span[@class="inline-btn user-button"]'
        },
    }
    sport: str = 'Baseball'
    months: list = ['Jan', 'Feb', 'Mar', 'April', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                    'Nov', 'Dec']

    # set restrictions to specific leagues
    euro_year: bool = False
    wc_year: bool = False
    champs_season: bool = True
    europa_season: bool = True

    # max workers for threading job
    max_workers_internal: int = 10
    max_match_workers: int = 2
    matches_per_league_major: int = 25
    matches_per_league_minor: int = 20
    max_upcoming_matches: int = 50
    max_scrolling_iter: int = 10

    # basketball settings
    selectors: dict = {
            'fixtures': {
                'by': By.TAG_NAME,
                'selector': 'a'
            },
            'home/away': {
                'books': {
                    'by': By.XPATH,
                    'selector': "//a/p[@class='height-content max-mm:hidden pl-4']"
                },
                'odds': {
                    'by': By.XPATH,
                    'selector': '//div[@class="flex flex-row items-center gap-[3px]"]/p[@class="height-content"]'
                    '| //div[@class="flex flex-row items-center gap-[3px]"]/p[@class="height-content line-through"]'
                    '| //div[@class="flex flex-row items-center gap-[3px]"]/a[@class="visible min-mt:!'
                                'hidden text-black-main underline"]'
                },
                'button': {
                    'by': By.XPATH,
                    'selector1': '//span[@class="flex"]/div[contains(text(), "Home/Away")]',
                    'selector2': '//span[@class="flex"]/div[contains(text(), "1X2")]'
                }
            },
            'over/under': {
                'books': {
                    'by': By.XPATH,
                    'selector': "//a/p[@class='height-content max-mm:hidden pl-4']"
                },
                'odds': {
                    'by': By.XPATH,
                    'selector': '//div/p[@class="height-content"] | //div/p[@class="height-content line-through"]'
                                '| //div/a[@class="hidden min-mt:!flex text-color-black underline"]'
                }
            },
            'moneylines': {
                'weights': {
                    'by': By.XPATH,
                    'selector': '//div/p[@class="ml-auto pr-3 text-xs font-normal"]'
                },
                'odds': {
                    'by': By.XPATH,
                    'selector': '//div/p[@class="max-sm:!hidden"]'
                },
                'alternative': {
                    'by': By.XPATH,
                    'selector': '//div[@class="flex flex-col max-sm:!hidden items-center justify-center gap-1 bo'
                                'rder-gray-medium min-w-[60px] max-sm:min-w-[55px] text-sm text-[#2F2F2F] border-l"]'
                }
            }
        }

    # set the chrome-version
    uc.TARGET_VERSION = 113

    # odds group counter
    odds_group_n: dict = {
        'Basketball': 2,
        'Football': 3,
        'AmericanFootball': 2,
        'Baseball': 2,
        'Tennis': 2,
        'Hockey': 2
    }

    # general url key for each sport
    general_url_key: dict = {
        'Basketball': 1,
        'Football': 2,
        'Baseball': 1,
        'Hockey': 2,
        'Tennis': 2,
        'AmericanFootball': 1
    }


class Utilities:
    """utility class with general functionalities"""

    def __init__(self) -> None:
        self.settings: Any = SelectorSettings()
        self.sports_accepted: List[str] = ['Basketball', 'Hockey', 'Baseball', 'Football', 'Tennis',
                                           'AmericanFootball']
        self.proxy_path: str = '/content/drive/MyDrive/baseball/proxies.txt'
        self.user_agent_path: str = '/content/drive/MyDrive/baseball/user_agents.txt'
        self.login_path: str = '/login'

    def correct_date(self, date: str):
        """post-process the date string pulled from the scraper"""

        # create a new mapping for the months of the year
        new_month_mapping = dict(zip(self.settings.months, ['January', 'February', 'March', 'April', 'May', 'June',
                                                'July', 'August', 'September', 'October', 'November', 'December']))

        # iterate through each key value pair and conduct the replace operation
        for key, value in new_month_mapping.items():
            date = date.replace(key, value)
        return date

    @staticmethod
    def unique_append(container: list, novel_items: list):
        """uniquely add items to the container"""

        # iterate through the novel items and check if they exist in the new container
        for _data in novel_items:
            if _data not in container:
                container.append(_data)

        return container

    @staticmethod
    def format_tournament_name(tournament_name: str) -> str:
        """format the name of a tournament to look more professional"""

        # error handling blob
        try:
            # remove the hyphens
            tournament_name_processed = tournament_name.replace('-', ' ')

            # if it is a single word then capitalize the entire tournament name
            if len(tournament_name_processed.split(' ')) == 1:
                return tournament_name_processed.upper()

            # capitalize each letter in the name
            tournament_name_processed: str = ' '.join([i.capitalize() for i in tournament_name_processed.split(' ')])

            return tournament_name_processed
        except (TypeError, ValueError):
            # output the original if an error is encountered
            return tournament_name

    @staticmethod
    def format_tournament_country(tournament_name: str) -> str:
        """format the country of origin for a tournament and handle exceptions"""


class PostProcessing:

    def __init__(self, data: list, match_links: list):
        """initialize post-processing objects"""
        self.data = data
        self.match_links = match_links
        self._reformatting_idx: dict = {
            list: self._list_branch,
            dict: self._dict_branch
        }

    def _convert_odds_numerical(self, odds_arr: list):
        """convert odds from any type into numerical form"""

        # define storage container
        container: list = []

        # iterate through the array and handle any errors arising from type errors
        for odd in odds_arr:
            try:
                container.append(float(odd))
            except (TypeError, ValueError):
                container.append(odd)
        return container

    def _remove_empty_odds(self, moneyline_mapping: dict) -> dict:
        """remove empty odd pairs from the odds collection"""

        # create container for new mapping
        new_mapping: dict = {}

        # iterate through mapping and remove missing odd pairs
        for key, value in moneyline_mapping.items():
            if not ('-' in value or None in value):  # check for missing odds
                new_mapping[key] = value

        return new_mapping

    def _dict_branch(self, moneyline: str, specific_odd_blob: dict) -> list:
        """option when type of input is a list such as over under"""

        # define output container for odds mappings
        container: list = []

        # iterate through odds mapping and reformat
        for key, value in specific_odd_blob.items():
            # define sub container for temporary data storage
            sub_container: dict = {}

            # populate the sub container instance
            sub_container['sportsbook'] = key

            sub_container['values'] = self._convert_odds_numerical(value)

            container.append({'odds': sub_container})
        return container

    def _list_branch(self, moneyline: str, specific_odd_blob: list) -> list:
        """option when type of input is a list such as over under"""

        # define output container
        container: list = []

        # iterate over internal moneylines and the process the odd blob from there
        for packet in specific_odd_blob:
            # define a sub container instance
            sub_container: dict = {}

            # extract the moneyline and place data into sub container
            try:
                specific_moneyline = list(packet.keys())[0]
            except IndexError:
                container.append(sub_container)
                continue

            # add data to the sub container
            sub_container['values'], sub_container['line'] = self._dict_branch(moneyline, packet[specific_moneyline]), \
                                                             specific_moneyline

            # add the sub container to the general data container
            container.append(sub_container)
        return container

    def _reformat_inner_mapping(self, moneyline: str, specific_odd_blob: Union[dict, list]) -> list:
        """reformat the inner dictionary to support query keywords as key value pairs"""

        # get the type of the odds blob
        odd_type: Any = type(specific_odd_blob)

        # output the appropriate results based on the type of the data blob
        return self._reformatting_idx[odd_type](moneyline, specific_odd_blob)

    # noinspection PyTypeChecker
    def _compute_missing_odds(self, odds_values: list) -> List[Any]:
        """find the missing odds value"""

        # define the container for the odds to calculate the mean
        odds_container: list = []
        container: list = []

        # iterate through each odd value and add the numerical ones
        for item in odds_values:
            try:
                odds_container.append(item['Odds']['values'])
            except KeyError:
                continue

        # test dimension size
        try:
            dim_size: int = len(odds_container[0])
        except IndexError:
            return odds_container

        # create dataframe and impute all the missing values
        df = pd.DataFrame(data=odds_container, columns=[f'cat{i}' for i in range(dim_size)])

        # replace all missing values and impute them with the mean of the column
        df = df.replace('', nan)
        for col in df.columns:
            df[col].fillna(value=round(df[col].mean(), 2), inplace=True)
        df.fillna('', inplace=True)

        # iterate through each and add the value pair
        for odds_line, new_values in zip(odds_values, df.values.tolist()):
            try:
                odds_line['Odds']['values'] = new_values
                container.append(odds_line)
            except KeyError:
                continue

        return container

    def _fill_odds_gaps(self, data: list) -> list:
        """fill in gaps in the odds data"""

        # define the output container
        container: list = []

        # iterate through data and fetch each match
        for match in data:
            # define the match odds
            match_odds: dict = match['Odds']

            # iterate through each odd category
            for odds_category, odds_packet in match_odds.items():
                # check the type of odds either home away or over under
                if odds_category == 'home/away':
                    match['Odds'][odds_category] = self._compute_missing_odds(odds_packet)
                else:
                    # iterate through each specific line to apply the procedure
                    for idx, odds_line in enumerate(odds_packet):
                        try:
                            try:
                                match['odds'][odds_category][idx] = self._compute_missing_odds(odds_line['values'])[0]
                            except IndexError:
                                try:
                                    match['odds'][odds_category][idx] = self._compute_missing_odds(odds_line['values'])
                                except (IndexError, KeyError):
                                    pprint(match)
                        except KeyError:
                            # replace with empty array
                            match['odds'][odds_category] = []

            # add the match data to the container
            container.append(match)

        return container

    def process_scraped_results_db(self) -> list:
        """process the scraped results to be easily queryable"""

        # create container
        container: list = []

        # iterate through each match** in the data blob
        for match_data, overview in zip(self.data, self.match_links):
            # define sub container
            sub_container: dict = {}

            # add match overviews
            sub_container['OpponentA'], sub_container['OpponentB'], sub_container['Date'], sub_container['Time'] = \
                overview['OpponentA'], overview['OpponentB'], overview['Date'], overview['Time']
            sub_container['Sport'], sub_container['Tournament'] = overview['Sport'], overview['Tournament']

            # reformat the odds data
            temp_match_data: dict = {}
            for packet in match_data:
                for key, value in packet.items():
                    temp_match_data[key] = value

            # run odds data algorithms uppercase id
            sub_container['Odds'] = {key: self._reformat_inner_mapping(key, value) for key, value in
                                     temp_match_data.items()}

            # container to general storage
            container.append(sub_container)

        # output with corrected values
        return container
        #return self._fill_odds_gaps(container)


class ScraperBase(Utilities):
    """base functions for the web scraper"""

    def __init__(self, configurations: dict):
        super().__init__() # add the superclass call
        # define the configurations as a hash map
        self.configurations: Dict = configurations

        # instantiate settings model and set api keys for GPT-3.5
        openai.api_key = self.configurations['keys']['gpt_api_key']

        # factory method for parsing scraper results
        self.parsing_index: dict = {
            'ruleHA': RuleParserHA(),
            'HC/OU': RuleParserMNL(),
            'ruleHC': RuleParserMNL(),
            'gpt': GPTParser()
        }

    def read_proxy_file(self) -> List[str]:
        """read proxy file"""
        with open(self.proxy_path, 'r') as proxies:
            proxy_arr = proxies.readlines()
            # process the new lines on each proxy address
            proxy_arr: List[str] = [i.replace('\n', '') for i in proxy_arr]
        return proxy_arr

    def read_user_agent_file(self) -> List[str]:
        """read user agent file"""
        with open(self.user_agent_path, 'r') as agents:
            agent_arr = agents.readlines()
            # process the new lines on each user agent
            agent_arr: List[str] = [i.replace('\n', '') for i in agent_arr]
        return agent_arr

    @staticmethod
    def approve_link(link: str, sport: str) -> bool:
        """check if link is accurate"""

        # check if sport is accepted
        if len(link.split('/'))>7 and sport.lower() in link and 'results' not in link and 'standings' not in link and \
                'outrights/' not in link and '//#' not in link:
            return True
        elif sport.lower().replace('_', '') == 'americanfootball' and len(link.split('/'))>=8 and '#' not in link and \
                'outrights' not in link and 'redirect' not in link and 'standings' not in link and \
                'results' not in link:
            return True
        else:
            return False

    def create_driver_option(self) -> Options:
        """create options instance for each new chromedriver instance"""
        options: Options = Options()
        options.add_argument('--incognito')
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-application-cache")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-setuid-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument('--headless=new')
        return options

    def random_user_agent(self) -> str:
        """output a random user agent to use for scraper"""

        # list of user agents
        user_agents_arr: List[str] = self.read_user_agent_file()

        # get random user agent and output it
        return random.choice(user_agents_arr)

    def scrolling(self, driver: webdriver) -> None:
        """run scrolling functionality"""
        # scrolling functionality
        try:
            driver.execute_script("window.scrollTo(0, 900)") # try with moderate scroll first
            driver.find_element(by=self.settings.login['submit']['by'],
                                value=self.settings.login['submit']['selector']).click()
        except (ElementClickInterceptedException, NoSuchElementException):
            driver.execute_script("window.scrollTo(0, 1200)") # scroll all the way to the bottom otherwise
            try:
                driver.find_element(by=self.settings.login['submit']['by'],
                                    value=self.settings.login['submit']['selector']).click()
            except (ElementClickInterceptedException, NoSuchElementException):
                driver.execute_script("window.scrollTo(0, 800)") # scroll back if too far and ad is blocking
                driver.find_element(by=self.settings.login['submit']['by'],
                                    value=self.settings.login['submit']['selector']).click()

    def incremental_scrolling(self, driver: webdriver, magnitude: int) -> None:
        """incrementally scroll the webpage to fetch all results"""
        try:
            # scroll to a desired location on the webpage
            driver.execute_script(f"window.scrollTo(0, {magnitude})")
        except ElementClickInterceptedException:
            # handle an intercepted click error and simply do not perform action
            pass

    def accept_cookies(self, driver: webdriver) -> None:
        """accept cookies on website"""
        try:
            driver.find_element(by=By.ID, value='onetrust-accept-btn-handler').click()
        except NoSuchElementException:
            pass

    def form_completion(self, driver: webdriver) -> None:
        """complete form"""

        # accept cookies
        self.accept_cookies(driver)

        # fill in form for logging in
        driver.find_element(by=self.settings.login['username']['by'],
                            value=self.settings.login['username']['selector']
                            ).send_keys(self.configurations['keys']['oddsportal_username'])
        driver.find_element(by=self.settings.login['password']['by'],
                            value=self.settings.login['password']['selector']
                            ).send_keys(self.configurations['keys']['oddsportal_password'])
        driver.switch_to.window(driver.window_handles[0])

    def general_scrape_books_odds(self, driver: Any, target: str = 'home/away') -> Tuple[list, list]:
        """general scraping logic for all odds returns both books and odds"""

        try:
            books: Iterable = driver.find_elements(by=self.settings.selectors[target]['books']['by'],
                                          value=self.settings.selectors[target]['books']['selector'])
            over_under: Iterable = driver.find_elements(by=self.settings.selectors[target]['odds']['by'],
                                value=self.settings.selectors[target]['odds']['selector'])
        except (KeyError, IndexError, NoSuchElementException): # check for an index error or key error in the selector
            driver.refresh()
            sleep(3)
            try:
                books: Iterable = driver.find_elements(by=self.settings.selectors[target]['books']['by'],
                                                       value=self.settings.selectors[target]['books']['selector'])
                over_under: Iterable = driver.find_elements(by=self.settings.selectors[target]['odds']['by'],
                                                            value=self.settings.selectors[target]['odds']['selector'])
            except (KeyError, IndexError, NoSuchElementException):
                raise BrokenPipeError

        # post-process the books and over under odds
        books: list = [i.text for i in books]
        odds: list = [i.text for i in over_under]

        return (books, odds)

    def format_odds(self, moneyline: float, books: Union[Iterable, List[str]], odds: Iterable) -> dict:
        """formatting the output odds from the scraping process"""
        # find how many percentages there are in the list
        arr: Iterable = np.array(odds)

        # get number of results in list that contain the % symbol and shorten the odds array
        res = len(np.where(np.char.find(np.char.lower(arr), '%') > -1)[0])
        odds = odds[:-3 * res]
        odds_grouped = zip(*(iter(odds),) * 2)

        # organize odds by sports book
        packet: dict = {book: [float(odd[0]), float(odd[1])] for book, odd in zip(books, odds_grouped)}
        packet['moneyline'] = moneyline
        return packet

    def new_tab(self, driver: webdriver, url: str) -> None:
        """open internal tabs based on how many urls are necessary"""

        # open the necessary tabs
        driver.execute_script(f"window.open('{url}'); ") # open new tab with javascript
        self.incremental_scrolling(driver, 500)
        sleep(random.choice([i / 10 for i in range(10, 20)])) # wait for tab to load

    def scrape_moneylines(self, driver: webdriver) -> List[list]:
        """scrape the moneylines from the website both handicapped and over-under"""
        # scrape the moneyline odds
        moneylines_odds: Iterable = driver.find_elements(by=self.settings.selectors['moneylines']['odds']['by'],
                                                         value=self.settings.selectors['moneylines']['odds'][
                                                             'selector'])

        # scrape the market weights for each set of odds
        moneyline_weights: Iterable = driver.find_elements(by=self.settings.selectors['moneylines']['weights']['by'],
                                                           value=self.settings.selectors['moneylines']['weights'][
                                                               'selector'])

        # convert to text
        moneylines_odds: list = [i.text for i in moneylines_odds]
        moneyline_weights: list = [i.text for i in moneyline_weights]
        return [moneylines_odds, moneyline_weights]

    def alternative_moneyline_search(self, driver: webdriver) -> str:
        """alternative search mechanism for the moneyline in over/under or asian handicap scraping"""

        # search for data
        data = driver.find_elements(by=self.settings.selectors['moneylines']['alternative']['by'],
                             value=self.settings.selectors['moneylines']['alternative']['selector'])

        # convert to textual values and select the first
        data = [i.text for i in data]
        return data[0]


class Upcoming(ScraperBase):
    """get all upcoming matches for a specific sport"""

    def __init__(self, configurations: dict):
        super().__init__(configurations)
        self.base_link = self._create_base_link(self.settings.sport)
        self.processing_char_idx: dict = {
            'Football': self.process_odds_chars_football,
            'Basketball': self.process_odds_chars_basketball,
            'Baseball': self.processed_odds_chars_baseball,
            'AmericanFootball': self.processed_odds_chars_baseball,
            'Hockey': self.process_odds_chars_football,
            'Tennis': self.process_odds_chars_basketball
        }

    @staticmethod
    def _create_base_link(sport: str) -> str:
        """create the base link to scrape upcoming matches"""
        if sport != 'AmericanFootball':
            return f'https://www.oddsportal.com/{sport}/'
        else:
            return 'https://www.oddsportal.com/american-football/'

    def check_fixture_status(self, packet: str) -> int:
        """check if a month of the year is included in the sequence of scraped values"""
        flag: bool = False
        for month in self.settings.months:
            if month in packet:
                flag = True
                break

        # compute the status of the fixture
        if packet.split(' ').count(':') > 1:
            return 2
        elif flag:
            return 1
        else:
            return 0

    def find_month_idx(self, packet_seq: list) -> str:
        """find the location of the month object in the seq and process it"""

        # loop through months
        for month in self.settings.months:
            # loop through each packet in the data
            for idx, packet in enumerate(packet_seq):
                if month in packet and any(char.isdigit() for char in packet):
                    return packet
        return ''

    @staticmethod
    def process_odds_chars_football(processed_packet: list) -> List[Union[str, int, float]]:
        """process the character type odds"""

        # define container array
        arr: list = []

        # preprocess all the odds
        time = processed_packet[-8]
        opp_a = processed_packet[-7]
        opp_b = processed_packet[-5]
        arr.extend([time, opp_a, opp_b])  # add items to container array

        # handle the odds logic
        for i in [-4, -3, -2, -1]:
            try:
                arr.append(float(processed_packet[i]))  # try converting to float type
            except (TypeError, IndexError, ValueError):
                arr.append(processed_packet[i])  # otherwise leave as is to prevent data loss

        return arr

    @staticmethod
    def process_odds_chars_basketball(processed_packet: list) -> List[Union[str, int, float]]:
        """process the character type odds"""

        # define container array
        arr: list = []

        # process the packet a bit more for certain leagues - australia
        processed_packet = [i for i in processed_packet if i != 'FRO']

        # preprocess all the odds
        time = processed_packet[-7]
        if ':' not in time:
            raise ValueError # since game is not valid and likely already played if such pattern is encountered
        else:
            opp_a = processed_packet[-6]
            opp_b = processed_packet[-4]

        # add items to container array
        arr.extend([time, opp_a, opp_b])

        # handle the odds logic
        for i in [-3, -2, -1]:
            try:
                arr.append(float(processed_packet[i]))  # try converting to float type
            except (TypeError, IndexError, ValueError):
                arr.append(processed_packet[i])  # otherwise leave as is to prevent data loss

            if i == -3:
                arr.append('-')

        return arr

    @staticmethod
    def processed_odds_chars_baseball(processed_packet_orig: list) -> List[Union[str, int, float]]:
        """process the basketball odds characters"""

        # define output container
        container: list = []

        # refine the processed packet more to exclude potential advertisements
        processed_packet = [i for i in processed_packet_orig if (len(i.split(' ')) <= 3 or ',' in i)]
        processed_packet = [i for i in processed_packet if '–' not in i]

        # define a collection container for more complex filtering
        container_temp: list = []
        for idx, item in enumerate(processed_packet):
            if idx != len(processed_packet) - 1:
                if item not in [str(j) for j in range(50)]:
                    container_temp.append(item)
            else:
                container_temp.append(item)

        # get the time of the event first
        time: str = container_temp[-6]
        if ':' not in time:
            raise ValueError # since game is not valid and likely already played if such pattern is encountered
        else:
            opp_a = container_temp[-5]
            opp_b = container_temp[-4]

        # add items to container array
        container.extend([time, opp_a, opp_b])

        # process the odds of the event
        for i in [-3, -2, -1]:
            try:
                container.append(float(container_temp[i]))  # try converting to float type
            except (TypeError, IndexError, ValueError):
                container.append(container_temp[i])  # otherwise leave as is to prevent data loss

            # add the tie odds which are None in basketball
            if i == -3:
                container.append('-')

        return container

    def process_upcoming_results(self, blob: list, links, league: str):
        """process the blob of upcoming matches"""

        # create storage container for data and links
        container: List[Union[None, dict]] = []
        link_container: List[str] = []

        # loop through each of the data packets and check for fixture status and parsing operations
        current_date: str = ''
        for idx, (packet, link) in enumerate(zip(blob, links)):
            # process the packet of data regarding odds and general statistics
            processed_packet = [i for i in packet.replace('/', '').split('\n') if i != '']

            # match is ongoing or complete currently so skip
            if self.check_fixture_status(packet) == 2:
                continue
            # match has a date in it at the top of the section
            elif self.check_fixture_status(packet) == 1:
                # set the current month data
                date_info = self.find_month_idx(processed_packet).split(', ')[-1].split(' - ')[0]
                current_date = current_date if date_info == '' else date_info

            # extract data for team names, odds, and market width from the scraped data
            try:
                time, opp_a, opp_b, a_odds, tie_odds, b_odds, market_width = self.processing_char_idx[
                    self.settings.sport](processed_packet)
            except ValueError:
                continue

            # add data to containers in proper format
            container.append({'Date': self.correct_date(current_date), 'Time': time, 'OpponentA': opp_a,
                              'OpponentB': opp_b,  'OddsA': a_odds, 'OddsTie': tie_odds, 'OddsB': b_odds,
                              'Market_Size': market_width, 'Sport': self.settings.sport,
                              'Tournament': self.format_tournament_name(league)})
            link_container.append(link)

        return container, link_container

    def _select_league_counts(self, blob: list, league_name: str):
        """select matches and reduce count based on the league that they are in"""

        # filter for basketball leagues
        if self.settings.sport == 'Basketball':
            # select only major leagues for full upcoming matches
            if 'usa' in league_name.lower() or 'europe' in league_name.lower() or 'spain' in league_name.lower():
                return blob[:self.settings.matches_per_league_major]
            else:
                return blob[:self.settings.matches_per_league_minor]
        # filter for football leagues
        elif self.settings.sport == 'Football':
            if 'spain' in league_name.lower() or 'france' in league_name.lower() or 'england' \
                                     in league_name.lower() or 'italy' in league_name.lower():
                return blob[:self.settings.matches_per_league_major]
            else:
                return blob[:self.settings.matches_per_league_minor]
        # filter for tennis leagues
        elif self.settings.sport == 'Tennis':
            # only select atp tournaments as major
            if 'atp' in league_name.lower():
                return blob[:self.settings.matches_per_league_major]
            else:
                return blob[:self.settings.matches_per_league_minor]
        # filter for american football matches
        elif self.settings.sport == 'AmericanFootball':
            # output the entire array of matches
            return blob
        else:
            return blob[:self.settings.max_upcoming_matches]

    def _scrape_upcoming(self, driver: webdriver, scrolling: bool = True, max_scroll_retry: int = 2):
        """scrape the upcoming matches with scrolling functionality"""

        # conduct scraping operation
        data, links = [], []

        # loop through iterations of scrolling
        len_similarity_counter: int = 0
        prev_len: int = -1
        for _ in range(self.settings.max_scrolling_iter if scrolling else 1):
            #import uuid
            #driver.save_screenshot(f'{uuid.uuid4()}.png')
            # pull the odds data
            data_scraped = driver.find_elements(by=By.XPATH,
                                                value='//div[@class="eventRow flex w-full flex-col text-xs"]')
            # pull the links with the matches and fetch the href attribute
            links_scraped = driver.find_elements(by=By.XPATH, value="//a")
            links_scraped = [i.get_attribute('href') for i in links_scraped]  # extract links by href attribute

            # add new items to the main containers for data and links
            data.extend(data_scraped)  # expand the textual attributes of the odds data
            links.extend(links_scraped)

            if len(links_scraped) == prev_len: # check if the current length of the links is the same as the previous
                len_similarity_counter += 1
            else:
                len_similarity_counter = 0
            prev_len = len(links_scraped)  # keep track of the length of the links data

            # stop the list when the similarity counter is exceeded
            if len_similarity_counter == max_scroll_retry:
                break

            # scroll down the webpage to fetch more results
            if scrolling:
                self.incremental_scrolling(driver, 10000)
                sleep(1)
                self.incremental_scrolling(driver, -2000)
                sleep(1)

        # create links data ready to process by removing duplicates while preserving order
        links = list(dict.fromkeys([i for i in links if self.approve_link(i, self.settings.sport)]))
        data = list(dict.fromkeys([i.text for i in data]))

        return data, links

    def upcoming_matches(self, driver: webdriver, league_indicator: str):
        """get the upcoming matches for a particular sport"""

        # accept cookies
        self.accept_cookies(driver)

        # fetch the scraped data from the private scraping method while allowing scraping
        data, links = self._scrape_upcoming(driver, scrolling=True)

        if len(data) == 0 or len(links) == 0:
            # reload the page first
            driver.refresh()
            sleep(2)
            data, links = self._scrape_upcoming(driver, scrolling=True)

        print(data)
        print(links)
        print('\n')

        # preprocess the scraped data
        data, links = self.process_upcoming_results(data, links, league_indicator)
        for idx, link in enumerate(links):
            data[idx]['link'] = link

        # shorten the amount of matches coming by league
        data = self._select_league_counts(data, league_name=league_indicator)

        # close driver and return
        return {league_indicator.replace('/', '-'): data} # data package

    def _reduce_leagues(self, leagues: list):
        """reduce the leagues in available to scraper to optimize load"""

        # remove rarer leagues that are no as regular
        if self.settings.sport == 'football':
            if self.settings.euro_year:
                leagues.remove('europe/euro-2024/')
            elif self.settings.wc_year:
                leagues.remove('world/world-cup/')
            elif self.settings.champ_season:
                leagues.remove('europe/champions-league/')
            elif self.settings.europa_season:
                leagues.remove('europe/europa-league/')
        return leagues

    def facade(self, leagues: list):
        """scrape results from all leagues and return a response packet with each as a dict key"""

        # create container to store results
        container: list = []

        # set up the scraper
        options = self.create_driver_option()
        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": self.random_user_agent()})

        # call the web driver to a particular site
        links: list = [self.base_link.lower() + league_indicator for league_indicator in leagues]
        with ThreadPoolExecutor(max_workers=len(links)) as executor:
            executor.map(self.new_tab, [driver for _ in range(len(links))], links)
        print(driver.window_handles)
        # iterate through open tabs and scrape upcoming matches from each one
        for window in driver.window_handles[::-1]: # iterate through windows in reverse
            # switch the driver tab to the next random one (no particular order)
            driver.switch_to.window(window)
            print(driver.current_url.lower())
            # skip over the welcome tab in zxschrome
            if 'welcome' not in driver.current_url.lower() and 'data:' not in driver.current_url:
                try:
                    full_data = self.upcoming_matches(driver, driver.current_url.split('/')[-2])
                    container.append(full_data)
                except IndexError:
                    continue

        driver.quit() # close the driver

        for league in container:
            print(list(league.keys())[0], len(league[list(league.keys())[0]]))

        return container


class Scraper(ScraperBase):
    """main scraper function collection"""

    def __init__(self, configurations: dict):
        super().__init__(configurations)

        # define basketball base link for scraping process
        self.base_link: str = f'https://www.oddsportal.com/{self.settings.sport}/'
        self.upcoming = Upcoming(configurations) # create object for upcoming matches

        # scraper index general level
        self.scraper_index: dict = {
            'home/away': self._home_away,
            'over/under': self.pull_lines,
            'asian_handicap': self.pull_lines,
        }

        # scraper index for internal granular level tasks (ex. specific moneyline)
        self.internal_scraper_index: dict = {
            'home/away': self._internal_HA_scraper,
            'over/under': self._internal_scraper,
            'asian_handicap': self._internal_scraper
        }

    def _clean_fixtures_data(self, data_dump: Iterable) -> List[Any]: # links for each match
        """clean up fixtures data and store in list container for export"""

        # define container for each game link
        game_container: list = []

        for game in data_dump:
            # get the href link data for each match
            game_link = game.get_attribute('href')
            # check if the link is valid for basketball
            if self.approve_link(game_link, 'basketball'):
                game_container.append(game_link)

        # remove duplicates from container with set operator
        game_container = list(set(game_container))
        return game_container

    def _create_game_links(self) -> List[str]:
        """create the links for each league with collection of upcoming matches"""

        # create container for all links for basketball matches
        link_container: List[str] = []

        # iterate through each league and create unique link for match page
        for league_id in self.configurations['leagues'][self.settings.sport]:
            # create url
            link_container.append(f'{self.base_link}{league_id}')
        return link_container

    def extract_fixtures(self, league_link: str):
        """extract games from each page with new webdriver instance"""

        # create options menu and create driver instance with unique user agent
        options = self.create_driver_option()
        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": self.random_user_agent()})

        # get fixtures page and set the window panel
        driver.get(league_link)
        driver.switch_to.window(driver.window_handles[0])

        # accept cookies
        driver.find_element(by=By.ID, value='onetrust-accept-btn-handler').click()

        # scrape matches data
        games = driver.find_elements(by=self.settings.selectors['fixtures']['id'],
                                     value=self.settings.selectors['fixtures']['selector'])
        games = self._clean_fixtures_data(games)

        # remove the driver and output games container
        driver.quit()
        return games

    def _home_away(self, driver: Any) -> list:
        """run the scraper for each individual match"""

        # run the web driver for home/away odds and their corresponding books
        sleep(10)
        self.incremental_scrolling(driver, 1000) # scroll halfway down the page to make results visible
        home_away: Iterable = driver.find_elements(by=self.settings.selectors['home/away']['odds']['by'],
                            value=self.settings.selectors['home/away']['odds']['selector'])
        home_away_books: Iterable = driver.find_elements(by=self.settings.selectors['home/away']['books']['by'],
                                            value=self.settings.selectors['home/away']['books']['selector'])
        home_away = [i.text for i in home_away] # convert both data packets to text
        home_away_books = [i.text for i in home_away_books]

        # process the odds through parsing techniques
        return self.parsing_index['ruleHA'].parse(home_away, books=home_away_books, odds_type='home/away',
                                    odds_group_n=self.settings.odds_group_n[self.settings.sport])

    def _map_window_url(self, driver: webdriver) -> dict:
        """map the window ids to the urls"""

        # create storage container for url window mapping
        container: dict = {}

        # iterate through all windows and record urls in each tab
        for window in driver.window_handles:
            driver.switch_to.window(window) # switch tabs for each window id
            container[window] = driver.current_url # record the mapping
        return container

    def _internal_scraper(self, driver: webdriver, url: str) -> dict:
        """internal scraping for over/under statistics"""

        # scroll down the page to insure all results are visible
        self.incremental_scrolling(driver, 750)

        try:
            books, over_under_odds = self.general_scrape_books_odds(driver, target='over/under')
        except:
            # retry the scraping process if failed
            books, over_under_odds = [], [] # go for the over under

        # create output package
        if url.endswith(';1') or url.endswith(';0'):
            try:
                return {self.alternative_moneyline_search(driver): self.parsing_index['HC/OU'].parse(books,
                                                                                    odds=over_under_odds, role=2)}
            except IndexError:
                return {}
        else:
            return {url.split(';')[-2]: self.parsing_index['HC/OU'].parse(books, odds=over_under_odds, role=2)}

    # NOT IN USE
    def _internal_HA_scraper(self, driver: webdriver, url: str) -> dict:
        """internal scraping functionality for asian handicap"""
        # get the link
        driver.get(url)
        sleep(random.choice([i / 10 for i in range(20, 30)])) # include randomness to wait time

        # get the elements needed
        try:
            # get data from parent class general scraping method
            books, home_away_odds = self.general_scrape_books_odds(driver, target='home/away')
        except:
            # retry scraping by getting url agan
            driver.get(url)
            sleep(random.choice([i / 10 for i in range(20, 30)])) # include degree of randomness to wait time

            # get data from parent class general scraping method
            books, home_away_odds = self.general_scrape_books_odds(driver, target='home/away')

        # alternative handicap searching mechanism
        if url.endswith(';1') or url.endswith(';0'):
            return {self.alternative_moneyline_search(driver): self.parsing_index['HC/OU'].parse(books,
                                                                                                 odds=home_away_odds)}
        else:
            return {url.split(';')[0]: self.parsing_index['HC/OU'].parse(books, odds=home_away_odds)}

    def pull_lines(self, match_link: str, method: str) -> dict: # FULL METHOD <----
        """run scraper to find the over under odds per match"""

        # create container for full data
        full_container: dict = {}

        # set up the scraping architecture for stealth selenium
        options = self.create_driver_option()
        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": self.random_user_agent()})

        # get the login page to the website
        driver.get(f'https://www.oddsportal.com{self.login_path}')
        driver.maximize_window()
        sleep(10)
        # fill out the login form
        try:
            self.form_completion(driver)
        except:
            driver.refresh()
            sleep(20)
            self.form_completion(driver)
            driver.save_screenshot("errorformcompletion.png")
            return {}
            #self.pull_lines(match_link, method)

        # do the scrolling action
        try:
            self.scrolling(driver)
        except:
            sleep(2)
            try:
                self.scrolling(driver)
            except:
                driver.refresh()
                try:
                    self.scrolling(driver)
                except NoSuchElementException:
                    driver.save_screenshot("step2.png")
                    return {}

        # wait for match page to load
        sleep(random.choice([i / 10 for i in range(20, 30)]))

        # define method (handicaps or over under)
        specified_match_link: str = match_link + f'#over-under;{self.settings.general_url_key[self.settings.sport]}' \
            f'' if method == 'over/under' else match_link + \
                                               f'#ah;{self.settings.general_url_key[self.settings.sport]}'
        print(specified_match_link)

        # get the match link in question
        driver.get(specified_match_link)

        # explicit wait for the webdriver to find the loaded elements
        sleep(random.choice([i / 10 for i in range(20, 30)]))
        sleep(10)
        driver.save_screenshot("step4.png")

        # get the moneylines
        moneylines_odds, moneyline_weights = self.scrape_moneylines(driver)

        # get each of the money lines from the match page
        moneylines_odds, moneyline_weights = self.scrape_moneylines(driver)

        if len(moneylines_odds) == 0: # try again if failed
            sleep(2)
            moneylines_odds, moneyline_weights = self.scrape_moneylines(driver)

        # decide which parsing technique to use based on the method involved - handicap or over under
        data, lines = self.parsing_index['HC/OU'].parse(moneylines_odds, weights=moneyline_weights, role=1,
                    base_link=match_link, method=method, url_key=self.settings.general_url_key[self.settings.sport])
        sleep(10)
        print(lines)
        # jump to home/away page to check outright odds
        if method == 'over/under':
            try:
                driver.find_element(by=self.settings.selectors['home/away']['button']['by'],
                                value=self.settings.selectors['home/away']['button']['selector1']).click()
            except NoSuchElementException:
                try:
                    driver.find_element(by=self.settings.selectors['home/away']['button']['by'],
                                    value=self.settings.selectors['home/away']['button']['selector2']).click()
                except NoSuchElementException:
                    driver.refresh()
                    sleep(3)
                    try:
                        driver.find_element(by=self.settings.selectors['home/away']['button']['by'],
                                        value=self.settings.selectors['home/away']['button']['selector1']).click()
                    except NoSuchElementException:
                        print("No Data Available", match_link)

            sleep(15)
            # home/away odds
            driver.save_screenshot("step3.png")
            home_away_data: list = self._home_away(driver)
            full_container['home/away'] = home_away_data

        # threading ability
        drivers: list = [driver for _ in range(len(data))]

        # check if no data is present
        if len(data) == 0:
            full_container[method] = [self.internal_scraper_index[method](driver, driver.current_url)]
            return full_container

        # threading to open each tab
        with ThreadPoolExecutor(max_workers=max(len(data), 1)) as executor:
            executor.map(self.new_tab, drivers, data)

        sleep(30)

        # iteration for lines data extraction from each moneyline link
        lines_data: list = []
        for window in driver.window_handles[::-1]: # iterate through windows in reverse
            driver.switch_to.window(window)
            if ('over-under' in driver.current_url or '#ah' in driver.current_url) \
                    and driver.current_url != specified_match_link:
                driver.switch_to.window(window)
                lines_data.append(self.internal_scraper_index[method](driver, driver.current_url))
            sleep(1)
        full_container[method] = lines_data # add data to the full container

        # stop the webdriver
        driver.quit()

        # return response packet
        return full_container

    def internal_threaded_lines(self, match_link: str, additionals: dict) -> List[Any]:
        """threaded method for pulling lines concurrently"""

        # container for data
        container: List[Any] = []

        # threading parameters
        match_links: List[Any] = [match_link for _ in range(2)]
        methods: List[str] = ['over/under', 'asian_handicap']

        # threading functionality
        with ThreadPoolExecutor(max_workers=len(methods)) as executor:
            results: Iterable = executor.map(self.pull_lines, match_links, methods)
            for result in results:
                container.append(result)

        return [container, additionals]

    def _upload_dataset(self, data: list, links: list) -> None:
        """upload results to the dataset"""
        post_processing = PostProcessing(data, links)
        data = post_processing.process_scraped_results_db()

    def _process_upload(self, data: list, match_links: list, db: Any, Processing: Callable) -> None:
        """process and upload the results to the database in the backend"""

        # create instance of post processing class
        post_processing = Processing(data, match_links)

        # process the data before storing in the database
        data = post_processing.process_scraped_results_db()

        # print the data for error handling
        pprint(data)

        # upload results to the cloud database instance
        db.upload_results(data)

    def facade(self, leagues: list, db: Any, Processing: Callable) -> None:
        """run the facade pattern for the basketball scraper class"""

        # get the upcoming matches
        upcoming_matches: list = self.upcoming.facade(leagues)

        # get all the information for upcoming matches
        upcoming_matches_event: list = []
        for league in upcoming_matches:
            # make sure the links are all lower case
            upcoming_matches_event.extend(i for i in league[list(league.keys())[0]])

        # get the upcoming match links
        upcoming_match_links: list = []
        for league in upcoming_matches:
            # make sure the links are all lower case
            upcoming_match_links.extend(i['link'] for i in league[list(league.keys())[0]])

        # group the links to thread over them
        links_grouped = [upcoming_match_links[i:i+self.settings.max_match_workers] for i in
                         range(0, len(upcoming_match_links), self.settings.max_match_workers)]
        upcoming_matches_grouped = [upcoming_matches_event[i:i+self.settings.max_match_workers] for i in
                         range(0, len(upcoming_matches_event), self.settings.max_match_workers)]

        # loop through each link group
        for link_group, match_group in tqdm(zip(links_grouped, upcoming_matches_grouped)):
            # store the results here
            container: list = []
            matches: list = []

            # run threading operation across match links
            with ThreadPoolExecutor(max_workers=self.settings.max_match_workers) as executor:
                # error handling block
                try:
                    results: Iterable = executor.map(self.internal_threaded_lines, link_group, match_group)
                except:
                    try:
                        results: Iterable = executor.map(self.internal_threaded_lines, link_group, match_group)
                    except:
                        continue
                for result in results: # iterate through threaded results
                    container.append(result[0])
                    matches.append(result[1])

            # process and upload to the database
            self._process_upload(container, matches, db, Processing)


# manager function
def manager(configs: dict, timeout: int = 20, wait_freq: int = 8):
    """manager function for scraper"""

    # create class objects
    scraper = Scraper(configurations=configs)
    db = ScraperDB('Sports', collection=scraper.settings.sport)

    # return the sport currently running
    print(configs['leagues'][scraper.settings.sport])

    # run infinite loop
    reg_timeout_counter: int = 0
    while True:
        # pull data from the scraper
        scraper.facade(configs['leagues'][scraper.settings.sport], db, PostProcessing)

        # activate the timeout
        sleep(timeout)
        reg_timeout_counter += 1

        # longer wait timeout
        if reg_timeout_counter % wait_freq == 0:
            sleep(1800)

# get the configurations from the file
with open(Path("/content/drive/MyDrive/baseball/configsBaseball.toml"), mode="rb") as fp:
    configs: dict = tomli.load(fp)

#manager(configs)
scraper = Scraper(configurations=configs)
scraper.pull_lines('https://www.oddsportal.com/baseball/usa/mlb/chicago-cubs-chicago-white-sox-Qms8L4jh/', method='over/under')