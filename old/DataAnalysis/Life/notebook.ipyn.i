{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Restarted tf (Python 3.10.15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fabb87b1-efd4-4e24-8aea-9c03e1c73b83",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Recency (months)  Frequency (times)  Monetary (c.c. blood)  Time (months)  \\\n",
      "0                 2                 50                  12500             98   \n",
      "1                 0                 13                   3250             28   \n",
      "2                 1                 16                   4000             35   \n",
      "3                 2                 20                   5000             45   \n",
      "4                 1                 24                   6000             77   \n",
      "\n",
      "   whether he/she donated blood in March 2007  \n",
      "0                                           1  \n",
      "1                                           1  \n",
      "2                                           1  \n",
      "3                                           1  \n",
      "4                                           0  \n"
     ]
    }
   ],
   "source": [
    "# Step 1: Read data\n",
    "import pandas as pd\n",
    "transfusion = pd.read_csv(\"transfusion.data\")\n",
    "print(transfusion.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "78fec62a-58f0-4d1f-9a76-4311540fc0f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 748 entries, 0 to 747\n",
      "Data columns (total 5 columns):\n",
      " #   Column                                      Non-Null Count  Dtype\n",
      "---  ------                                      --------------  -----\n",
      " 0   Recency (months)                            748 non-null    int64\n",
      " 1   Frequency (times)                           748 non-null    int64\n",
      " 2   Monetary (c.c. blood)                       748 non-null    int64\n",
      " 3   Time (months)                               748 non-null    int64\n",
      " 4   whether he/she donated blood in March 2007  748 non-null    int64\n",
      "dtypes: int64(5)\n",
      "memory usage: 29.3 KB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "# Step 2: Data info\n",
    "print(transfusion.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "46f8f688-553b-4a12-ba30-93f268bd885a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Recency (months)  Frequency (times)  Monetary (c.c. blood)  Time (months)  \\\n",
      "0                 2                 50                  12500             98   \n",
      "1                 0                 13                   3250             28   \n",
      "\n",
      "   target  \n",
      "0       1  \n",
      "1       1  \n"
     ]
    }
   ],
   "source": [
    "# Step 3: Rename target column\n",
    "transfusion.rename(columns={'whether he/she donated blood in March 2007': 'target'}, inplace=True)\n",
    "print(transfusion.head(2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4108c8a2-ec60-4935-97e7-e580dc82dd84",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "target\n",
      "0    0.762\n",
      "1    0.238\n",
      "Name: proportion, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# Step 4: Target distribution\n",
    "print(transfusion['target'].value_counts(normalize=True).round(3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "65247fc9-3277-4128-b1dc-1ae042ebb737",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 5: Initial split\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X = transfusion.drop('target', axis=1)\n",
    "y = transfusion['target']\n",
    "\n",
    "# Split into temp and final test set\n",
    "X_temp, X_reserva, y_temp, y_reserva = train_test_split(\n",
    "    X, y, \n",
    "    test_size=0.2, \n",
    "    random_state=42, \n",
    "    stratify=y\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9097fd52-46fa-408d-b796-457e06827316",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 6: Train-validation split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_temp, y_temp,\n",
    "    test_size=0.25,\n",
    "    random_state=42,\n",
    "    stratify=y_temp\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f23162f-8d96-43e6-91e0-37858c238baf",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/tpot/tpot_estimator/estimator.py:456: UserWarning: Both generations and max_time_mins are set. TPOT will terminate when the first condition is met.\n",
      "  warnings.warn(\"Both generations and max_time_mins are set. TPOT will terminate when the first condition is met.\")\n"
     ]
    },
    {
     "ename": "TimeoutError",
     "evalue": "No valid workers found",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTimeoutError\u001b[0m                              Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[7], line 20\u001b[0m\n\u001b[1;32m      8\u001b[0m \u001b[39m# Updated TPOT configuration\u001b[39;00m\n\u001b[1;32m      9\u001b[0m tpot \u001b[39m=\u001b[39m TPOTClassifier(\n\u001b[1;32m     10\u001b[0m     generations\u001b[39m=\u001b[39m\u001b[39m5\u001b[39m,\n\u001b[1;32m     11\u001b[0m     population_size\u001b[39m=\u001b[39m\u001b[39m20\u001b[39m,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m     17\u001b[0m     n_jobs\u001b[39m=\u001b[39m\u001b[39m-\u001b[39m\u001b[39m1\u001b[39m\n\u001b[1;32m     18\u001b[0m )\n\u001b[0;32m---> 20\u001b[0m tpot\u001b[39m.\u001b[39;49mfit(X_train, y_train)\n\u001b[1;32m     22\u001b[0m \u001b[39m# Evaluation\u001b[39;00m\n\u001b[1;32m     23\u001b[0m tpot_auc \u001b[39m=\u001b[39m roc_auc_score(y_test, tpot\u001b[39m.\u001b[39mpredict_proba(X_test)[:, \u001b[39m1\u001b[39m])\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/tpot/tpot_estimator/templates/tpottemplates.py:568\u001b[0m, in \u001b[0;36mTPOTClassifier.fit\u001b[0;34m(self, X, y)\u001b[0m\n\u001b[1;32m    540\u001b[0m     \u001b[39msuper\u001b[39m(TPOTClassifier,\u001b[39mself\u001b[39m)\u001b[39m.\u001b[39m\u001b[39m__init__\u001b[39m(\n\u001b[1;32m    541\u001b[0m         search_space\u001b[39m=\u001b[39msearch_space,\n\u001b[1;32m    542\u001b[0m         scorers\u001b[39m=\u001b[39m\u001b[39mself\u001b[39m\u001b[39m.\u001b[39mscorers, \n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    564\u001b[0m         random_state\u001b[39m=\u001b[39m\u001b[39mself\u001b[39m\u001b[39m.\u001b[39mrandom_state,\n\u001b[1;32m    565\u001b[0m         \u001b[39m*\u001b[39m\u001b[39m*\u001b[39m\u001b[39mself\u001b[39m\u001b[39m.\u001b[39mtpotestimator_kwargs)\n\u001b[1;32m    566\u001b[0m     \u001b[39mself\u001b[39m\u001b[39m.\u001b[39minitialized \u001b[39m=\u001b[39m \u001b[39mTrue\u001b[39;00m\n\u001b[0;32m--> 568\u001b[0m \u001b[39mreturn\u001b[39;00m \u001b[39msuper\u001b[39;49m()\u001b[39m.\u001b[39;49mfit(X,y)\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/tpot/tpot_estimator/estimator.py:663\u001b[0m, in \u001b[0;36mTPOTEstimator.fit\u001b[0;34m(self, X, y)\u001b[0m\n\u001b[1;32m    660\u001b[0m     evaluation_early_stop_steps \u001b[39m=\u001b[39m \u001b[39mNone\u001b[39;00m\n\u001b[1;32m    662\u001b[0m \u001b[39mif\u001b[39;00m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mscatter:\n\u001b[0;32m--> 663\u001b[0m     X_future \u001b[39m=\u001b[39m _client\u001b[39m.\u001b[39;49mscatter(X)\n\u001b[1;32m    664\u001b[0m     y_future \u001b[39m=\u001b[39m _client\u001b[39m.\u001b[39mscatter(y)\n\u001b[1;32m    665\u001b[0m \u001b[39melse\u001b[39;00m:\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/distributed/client.py:2784\u001b[0m, in \u001b[0;36mClient.scatter\u001b[0;34m(self, data, workers, broadcast, direct, hash, timeout, asynchronous)\u001b[0m\n\u001b[1;32m   2782\u001b[0m \u001b[39mexcept\u001b[39;00m \u001b[39mValueError\u001b[39;00m:\n\u001b[1;32m   2783\u001b[0m     local_worker \u001b[39m=\u001b[39m \u001b[39mNone\u001b[39;00m\n\u001b[0;32m-> 2784\u001b[0m \u001b[39mreturn\u001b[39;00m \u001b[39mself\u001b[39;49m\u001b[39m.\u001b[39;49msync(\n\u001b[1;32m   2785\u001b[0m     \u001b[39mself\u001b[39;49m\u001b[39m.\u001b[39;49m_scatter,\n\u001b[1;32m   2786\u001b[0m     data,\n\u001b[1;32m   2787\u001b[0m     workers\u001b[39m=\u001b[39;49mworkers,\n\u001b[1;32m   2788\u001b[0m     broadcast\u001b[39m=\u001b[39;49mbroadcast,\n\u001b[1;32m   2789\u001b[0m     direct\u001b[39m=\u001b[39;49mdirect,\n\u001b[1;32m   2790\u001b[0m     local_worker\u001b[39m=\u001b[39;49mlocal_worker,\n\u001b[1;32m   2791\u001b[0m     timeout\u001b[39m=\u001b[39;49mtimeout,\n\u001b[1;32m   2792\u001b[0m     asynchronous\u001b[39m=\u001b[39;49masynchronous,\n\u001b[1;32m   2793\u001b[0m     \u001b[39mhash\u001b[39;49m\u001b[39m=\u001b[39;49m\u001b[39mhash\u001b[39;49m,\n\u001b[1;32m   2794\u001b[0m )\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/distributed/utils.py:363\u001b[0m, in \u001b[0;36mSyncMethodMixin.sync\u001b[0;34m(self, func, asynchronous, callback_timeout, *args, **kwargs)\u001b[0m\n\u001b[1;32m    361\u001b[0m     \u001b[39mreturn\u001b[39;00m future\n\u001b[1;32m    362\u001b[0m \u001b[39melse\u001b[39;00m:\n\u001b[0;32m--> 363\u001b[0m     \u001b[39mreturn\u001b[39;00m sync(\n\u001b[1;32m    364\u001b[0m         \u001b[39mself\u001b[39;49m\u001b[39m.\u001b[39;49mloop, func, \u001b[39m*\u001b[39;49margs, callback_timeout\u001b[39m=\u001b[39;49mcallback_timeout, \u001b[39m*\u001b[39;49m\u001b[39m*\u001b[39;49mkwargs\n\u001b[1;32m    365\u001b[0m     )\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/distributed/utils.py:439\u001b[0m, in \u001b[0;36msync\u001b[0;34m(loop, func, callback_timeout, *args, **kwargs)\u001b[0m\n\u001b[1;32m    436\u001b[0m         wait(\u001b[39m10\u001b[39m)\n\u001b[1;32m    438\u001b[0m \u001b[39mif\u001b[39;00m error \u001b[39mis\u001b[39;00m \u001b[39mnot\u001b[39;00m \u001b[39mNone\u001b[39;00m:\n\u001b[0;32m--> 439\u001b[0m     \u001b[39mraise\u001b[39;00m error\n\u001b[1;32m    440\u001b[0m \u001b[39melse\u001b[39;00m:\n\u001b[1;32m    441\u001b[0m     \u001b[39mreturn\u001b[39;00m result\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/distributed/utils.py:413\u001b[0m, in \u001b[0;36msync.<locals>.f\u001b[0;34m()\u001b[0m\n\u001b[1;32m    411\u001b[0m         awaitable \u001b[39m=\u001b[39m wait_for(awaitable, timeout)\n\u001b[1;32m    412\u001b[0m     future \u001b[39m=\u001b[39m asyncio\u001b[39m.\u001b[39mensure_future(awaitable)\n\u001b[0;32m--> 413\u001b[0m     result \u001b[39m=\u001b[39m \u001b[39myield\u001b[39;00m future\n\u001b[1;32m    414\u001b[0m \u001b[39mexcept\u001b[39;00m \u001b[39mException\u001b[39;00m \u001b[39mas\u001b[39;00m exception:\n\u001b[1;32m    415\u001b[0m     error \u001b[39m=\u001b[39m exception\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/tornado/gen.py:766\u001b[0m, in \u001b[0;36mRunner.run\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    764\u001b[0m \u001b[39mtry\u001b[39;00m:\n\u001b[1;32m    765\u001b[0m     \u001b[39mtry\u001b[39;00m:\n\u001b[0;32m--> 766\u001b[0m         value \u001b[39m=\u001b[39m future\u001b[39m.\u001b[39;49mresult()\n\u001b[1;32m    767\u001b[0m     \u001b[39mexcept\u001b[39;00m \u001b[39mException\u001b[39;00m \u001b[39mas\u001b[39;00m e:\n\u001b[1;32m    768\u001b[0m         \u001b[39m# Save the exception for later. It's important that\u001b[39;00m\n\u001b[1;32m    769\u001b[0m         \u001b[39m# gen.throw() not be called inside this try/except block\u001b[39;00m\n\u001b[1;32m    770\u001b[0m         \u001b[39m# because that makes sys.exc_info behave unexpectedly.\u001b[39;00m\n\u001b[1;32m    771\u001b[0m         exc: Optional[\u001b[39mException\u001b[39;00m] \u001b[39m=\u001b[39m e\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/distributed/client.py:2654\u001b[0m, in \u001b[0;36mClient._scatter\u001b[0;34m(self, data, workers, broadcast, direct, local_worker, timeout, hash)\u001b[0m\n\u001b[1;32m   2650\u001b[0m         \u001b[39mawait\u001b[39;00m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mscheduler\u001b[39m.\u001b[39mupdate_data(\n\u001b[1;32m   2651\u001b[0m             who_has\u001b[39m=\u001b[39mwho_has, nbytes\u001b[39m=\u001b[39mnbytes, client\u001b[39m=\u001b[39m\u001b[39mself\u001b[39m\u001b[39m.\u001b[39mid\n\u001b[1;32m   2652\u001b[0m         )\n\u001b[1;32m   2653\u001b[0m     \u001b[39melse\u001b[39;00m:\n\u001b[0;32m-> 2654\u001b[0m         \u001b[39mawait\u001b[39;00m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mscheduler\u001b[39m.\u001b[39mscatter(\n\u001b[1;32m   2655\u001b[0m             data\u001b[39m=\u001b[39mdata2,\n\u001b[1;32m   2656\u001b[0m             workers\u001b[39m=\u001b[39mworkers,\n\u001b[1;32m   2657\u001b[0m             client\u001b[39m=\u001b[39m\u001b[39mself\u001b[39m\u001b[39m.\u001b[39mid,\n\u001b[1;32m   2658\u001b[0m             broadcast\u001b[39m=\u001b[39mbroadcast,\n\u001b[1;32m   2659\u001b[0m             timeout\u001b[39m=\u001b[39mtimeout,\n\u001b[1;32m   2660\u001b[0m         )\n\u001b[1;32m   2662\u001b[0m out \u001b[39m=\u001b[39m {k: Future(k, \u001b[39mself\u001b[39m) \u001b[39mfor\u001b[39;00m k \u001b[39min\u001b[39;00m data}\n\u001b[1;32m   2663\u001b[0m \u001b[39mfor\u001b[39;00m key, typ \u001b[39min\u001b[39;00m types\u001b[39m.\u001b[39mitems():\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/distributed/core.py:1259\u001b[0m, in \u001b[0;36mPooledRPCCall.__getattr__.<locals>.send_recv_from_rpc\u001b[0;34m(**kwargs)\u001b[0m\n\u001b[1;32m   1257\u001b[0m prev_name, comm\u001b[39m.\u001b[39mname \u001b[39m=\u001b[39m comm\u001b[39m.\u001b[39mname, \u001b[39m\"\u001b[39m\u001b[39mConnectionPool.\u001b[39m\u001b[39m\"\u001b[39m \u001b[39m+\u001b[39m key\n\u001b[1;32m   1258\u001b[0m \u001b[39mtry\u001b[39;00m:\n\u001b[0;32m-> 1259\u001b[0m     \u001b[39mreturn\u001b[39;00m \u001b[39mawait\u001b[39;00m send_recv(comm\u001b[39m=\u001b[39mcomm, op\u001b[39m=\u001b[39mkey, \u001b[39m*\u001b[39m\u001b[39m*\u001b[39mkwargs)\n\u001b[1;32m   1260\u001b[0m \u001b[39mfinally\u001b[39;00m:\n\u001b[1;32m   1261\u001b[0m     \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mpool\u001b[39m.\u001b[39mreuse(\u001b[39mself\u001b[39m\u001b[39m.\u001b[39maddr, comm)\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/distributed/core.py:1043\u001b[0m, in \u001b[0;36msend_recv\u001b[0;34m(comm, reply, serializers, deserializers, **kwargs)\u001b[0m\n\u001b[1;32m   1041\u001b[0m     _, exc, tb \u001b[39m=\u001b[39m clean_exception(\u001b[39m*\u001b[39m\u001b[39m*\u001b[39mresponse)\n\u001b[1;32m   1042\u001b[0m     \u001b[39massert\u001b[39;00m exc\n\u001b[0;32m-> 1043\u001b[0m     \u001b[39mraise\u001b[39;00m exc\u001b[39m.\u001b[39mwith_traceback(tb)\n\u001b[1;32m   1044\u001b[0m \u001b[39melse\u001b[39;00m:\n\u001b[1;32m   1045\u001b[0m     \u001b[39mraise\u001b[39;00m \u001b[39mException\u001b[39;00m(response[\u001b[39m\"\u001b[39m\u001b[39mexception_text\u001b[39m\u001b[39m\"\u001b[39m])\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/distributed/core.py:834\u001b[0m, in \u001b[0;36m_handle_comm\u001b[0;34m()\u001b[0m\n\u001b[1;32m    832\u001b[0m     result \u001b[39m=\u001b[39m handler(\u001b[39m*\u001b[39m\u001b[39m*\u001b[39mmsg)\n\u001b[1;32m    833\u001b[0m \u001b[39mif\u001b[39;00m inspect\u001b[39m.\u001b[39miscoroutine(result):\n\u001b[0;32m--> 834\u001b[0m     result \u001b[39m=\u001b[39m \u001b[39mawait\u001b[39;00m result\n\u001b[1;32m    835\u001b[0m \u001b[39melif\u001b[39;00m inspect\u001b[39m.\u001b[39misawaitable(result):\n\u001b[1;32m    836\u001b[0m     \u001b[39mraise\u001b[39;00m \u001b[39mRuntimeError\u001b[39;00m(\n\u001b[1;32m    837\u001b[0m         \u001b[39mf\u001b[39m\u001b[39m\"\u001b[39m\u001b[39mComm handler returned unknown awaitable. Expected coroutine, instead got \u001b[39m\u001b[39m{\u001b[39;00m\u001b[39mtype\u001b[39m(result)\u001b[39m}\u001b[39;00m\u001b[39m\"\u001b[39m\n\u001b[1;32m    838\u001b[0m     )\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/distributed/scheduler.py:6363\u001b[0m, in \u001b[0;36mscatter\u001b[0;34m()\u001b[0m\n\u001b[1;32m   6361\u001b[0m         \u001b[39mbreak\u001b[39;00m\n\u001b[1;32m   6362\u001b[0m     \u001b[39mif\u001b[39;00m time() \u001b[39m>\u001b[39m start \u001b[39m+\u001b[39m timeout:\n\u001b[0;32m-> 6363\u001b[0m         \u001b[39mraise\u001b[39;00m \u001b[39mTimeoutError\u001b[39;00m(\u001b[39m\"\u001b[39m\u001b[39mNo valid workers found\u001b[39m\u001b[39m\"\u001b[39m)\n\u001b[1;32m   6364\u001b[0m     \u001b[39mawait\u001b[39;00m asyncio\u001b[39m.\u001b[39msleep(\u001b[39m0.1\u001b[39m)\n\u001b[1;32m   6366\u001b[0m \u001b[39massert\u001b[39;00m \u001b[39misinstance\u001b[39m(data, \u001b[39mdict\u001b[39m)\n",
      "\u001b[0;31mTimeoutError\u001b[0m: No valid workers found"
     ]
    }
   ],
   "source": [
    "# Step 7: TPOT setup (updated parameters)\n",
    "#import os\n",
    "#os.environ[\"TPOT_NO_DASK\"] = \"true\"\n",
    "from tpot import TPOTClassifier\n",
    "from sklearn.metrics import roc_auc_score\n",
    "\n",
    "# Updated TPOT configuration\n",
    "tpot = TPOTClassifier(\n",
    "    generations=5,\n",
    "    population_size=20,\n",
    "    cv=5,\n",
    "    scorers=['roc_auc'],  # Correct parameter name for scoring\n",
    "    scorers_weights=[1.0],  # Weight for the scorer\n",
    "    verbose=2,\n",
    "    random_state=42,\n",
    "    n_jobs=-1\n",
    ")\n",
    "\n",
    "tpot.fit(X_train, y_train)\n",
    "\n",
    "# Evaluation\n",
    "tpot_auc = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])\n",
    "print(f\"TPOT AUC: {tpot_auc:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "241b26e7-4177-4363-9de1-dfdbfb816a1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 8: Feature engineering\n",
    "import numpy as np\n",
    "\n",
    "# Log-transform high-variance feature\n",
    "col_to_normalize = X_train.var().idxmax()\n",
    "for df in [X_train, X_test, X_temp, X_reserva]:\n",
    "    df['monetary_log'] = np.log(df[col_to_normalize])\n",
    "    df.drop(col_to_normalize, axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b6567ed7-b540-4ec2-9327-18c3624ae25e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Baseline AUC: 0.7387\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/homebrew/Caskroom/miniconda/base/envs/tf/lib/python3.10/site-packages/sklearn/linear_model/_sag.py:349: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "# Step 9: Logistic regression baseline\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "logreg = LogisticRegression(solver='saga', max_iter=1000, random_state=42)\n",
    "logreg.fit(X_train, y_train)\n",
    "logreg_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])\n",
    "print(f\"Baseline AUC: {logreg_auc:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c5b1353-1020-483a-bea4-7ef3cf4f432b",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'logreg_cv' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[10], line 3\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[39m#%%\u001b[39;00m\n\u001b[1;32m      2\u001b[0m \u001b[39m# Step 11: Final evaluation on holdout set\u001b[39;00m\n\u001b[0;32m----> 3\u001b[0m final_model \u001b[39m=\u001b[39m logreg_cv\u001b[39m.\u001b[39mbest_estimator_\n\u001b[1;32m      4\u001b[0m final_model\u001b[39m.\u001b[39mfit(X_temp, y_temp)  \u001b[39m# Train on full temp data\u001b[39;00m\n\u001b[1;32m      5\u001b[0m reserva_auc \u001b[39m=\u001b[39m roc_auc_score(y_reserva, final_model\u001b[39m.\u001b[39mpredict_proba(X_reserva)[:, \u001b[39m1\u001b[39m])\n",
      "\u001b[0;31mNameError\u001b[0m: name 'logreg_cv' is not defined"
     ]
    }
   ],
   "source": [
    "# Step 11: Final evaluation on holdout set\n",
    "final_model = logreg_cv.best_estimator_\n",
    "final_model.fit(X_temp, y_temp)  # Train on full temp data\n",
    "reserva_auc = roc_auc_score(y_reserva, final_model.predict_proba(X_reserva)[:, 1])\n",
    "print(f\"Final Holdout AUC: {reserva_auc:.4f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "tf",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
