CREATE TABLE `qml_models` (
  `model_id` int(11) NOT NULL AUTO_INCREMENT,
  `cls` varchar(100) NOT NULL DEFAULT '',
  `params` varchar(4000) NOT NULL DEFAULT '',
  `descr` varchar(1000) NOT NULL DEFAULT '',
  `predict_fn` varchar(200) NOT NULL DEFAULT 'predict',
  `level` int(11) NOT NULL DEFAULT '1',
  PRIMARY KEY (`model_id`)
) ENGINE=InnoDB AUTO_INCREMENT=1000 DEFAULT CHARSET=utf8;

CREATE TABLE `qml_results` (
  `data_id` int(11) NOT NULL,
  `model_id` int(11) NOT NULL,
  `cv_score` decimal(18,10) DEFAULT NULL,
  `cv_time` decimal(18,1) DEFAULT NULL,
  `public_board_score` decimal(18,10) DEFAULT NULL,
  PRIMARY KEY (`data_id`,`model_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `qml_results_statistic` (
  `data_id` int(11) NOT NULL,
  `model_id` int(11) NOT NULL,
  `fold` int(11) NOT NULL,
  `seed` int(11) NOT NULL,
  `cv_score` decimal(18,10) NOT NULL,
  PRIMARY KEY (`data_id`,`model_id`,`fold`,`seed`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
