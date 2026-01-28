/*
 Navicat Premium Data Transfer

 Source Server         : local
 Source Server Type    : MySQL
 Source Server Version : 100432 (10.4.32-MariaDB)
 Source Host           : localhost:3306
 Source Schema         : user_matchmaking

 Target Server Type    : MySQL
 Target Server Version : 100432 (10.4.32-MariaDB)
 File Encoding         : 65001

 Date: 28/01/2026 21:28:02
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for user_pair_score
-- ----------------------------
DROP TABLE IF EXISTS `user_pair_score`;
CREATE TABLE `user_pair_score`  (
  `user_id_1` int NOT NULL,
  `user_id_2` int NOT NULL,
  `score` float NULL DEFAULT NULL,
  PRIMARY KEY (`user_id_1`, `user_id_2`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for user_traits
-- ----------------------------
DROP TABLE IF EXISTS `user_traits`;
CREATE TABLE `user_traits`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `t1` float NULL DEFAULT NULL,
  `t2` float NULL DEFAULT NULL,
  `t3` float NULL DEFAULT NULL,
  `t4` float NULL DEFAULT NULL,
  `t5` float NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 4001 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Procedure structure for generate_scores
-- ----------------------------
DROP PROCEDURE IF EXISTS `generate_scores`;
delimiter ;;
CREATE PROCEDURE `generate_scores`()
BEGIN
	INSERT INTO user_pair_score 
	SELECT
		a.id AS u1_id,
		b.id AS u2_id,
		match_score_traits(a.t1,a.t2,a.t3,a.t4,a.t5, b.t1,b.t2,b.t3,b.t4,b.t5) AS score
	FROM
	(
		SELECT id, t1,t2,t3,t4,t5
		FROM user_traits
		WHERE id >= (SELECT FLOOR(RAND() * (SELECT MAX(id) FROM user_traits)))
		ORDER BY id
		LIMIT 1000
	) a
	JOIN
	(
		SELECT id, t1,t2,t3,t4,t5
		FROM user_traits
		WHERE id >= (SELECT FLOOR(RAND() * (SELECT MAX(id) FROM user_traits)))
		ORDER BY id
		LIMIT 1000
	) b
		ON a.id < b.id
	LIMIT 100000;


END
;;
delimiter ;

-- ----------------------------
-- Function structure for match_score_traits
-- ----------------------------
DROP FUNCTION IF EXISTS `match_score_traits`;
delimiter ;;
CREATE FUNCTION `match_score_traits`(a1 DOUBLE, a2 DOUBLE, a3 DOUBLE, a4 DOUBLE, a5 DOUBLE,
  b1 DOUBLE, b2 DOUBLE, b3 DOUBLE, b4 DOUBLE, b5 DOUBLE)
 RETURNS double
  DETERMINISTIC
BEGIN
  -- Canonicalize so AB == BA
  DECLARE t1_1 DOUBLE; DECLARE t1_2 DOUBLE; DECLARE t1_3 DOUBLE; DECLARE t1_4 DOUBLE; DECLARE t1_5 DOUBLE;
  DECLARE t2_1 DOUBLE; DECLARE t2_2 DOUBLE; DECLARE t2_3 DOUBLE; DECLARE t2_4 DOUBLE; DECLARE t2_5 DOUBLE;

  -- closeness per trait in [0..1]
  DECLARE c1 DOUBLE; DECLARE c2 DOUBLE; DECLARE c3 DOUBLE; DECLARE c4 DOUBLE; DECLARE c5 DOUBLE;

  DECLARE minC DOUBLE;
  DECLARE geo DOUBLE;
  DECLARE synergy DOUBLE;
  DECLARE dealPenalty DOUBLE;
  DECLARE raw DOUBLE;
  DECLARE scoreBase DOUBLE;

  -- Assumed trait range ~ 1..100 (floats). Denominator is 99.
  -- If you use 0..100 instead, change denom to 100.0.
  DECLARE denom DOUBLE DEFAULT 100;

  -- 1) Lexicographic canonical ordering (numeric, not string)
  IF (a1 > b1)
     OR (a1 = b1 AND a2 > b2)
     OR (a1 = b1 AND a2 = b2 AND a3 > b3)
     OR (a1 = b1 AND a2 = b2 AND a3 = b3 AND a4 > b4)
     OR (a1 = b1 AND a2 = b2 AND a3 = b3 AND a4 = b4 AND a5 > b5)
  THEN
    SET t1_1=b1; SET t1_2=b2; SET t1_3=b3; SET t1_4=b4; SET t1_5=b5;
    SET t2_1=a1; SET t2_2=a2; SET t2_3=a3; SET t2_4=a4; SET t2_5=a5;
  ELSE
    SET t1_1=a1; SET t1_2=a2; SET t1_3=a3; SET t1_4=a4; SET t1_5=a5;
    SET t2_1=b1; SET t2_2=b2; SET t2_3=b3; SET t2_4=b4; SET t2_5=b5;
  END IF;

  -- 2) Closeness: 1=identical, 0=max different (normalized by denom)
  SET c1 = 1.0 - (ABS(t1_1 - t2_1) / denom);
  SET c2 = 1.0 - (ABS(t1_2 - t2_2) / denom);
  SET c3 = 1.0 - (ABS(t1_3 - t2_3) / denom);
  SET c4 = 1.0 - (ABS(t1_4 - t2_4) / denom);
  SET c5 = 1.0 - (ABS(t1_5 - t2_5) / denom);

  -- clamp to [0,1]
  SET c1 = LEAST(1.0, GREATEST(0.0, c1));
  SET c2 = LEAST(1.0, GREATEST(0.0, c2));
  SET c3 = LEAST(1.0, GREATEST(0.0, c3));
  SET c4 = LEAST(1.0, GREATEST(0.0, c4));
  SET c5 = LEAST(1.0, GREATEST(0.0, c5));

  SET minC = LEAST(c1, c2, c3, c4, c5);

  -- 3) Non-linear aggregation (spread + high scores possible)
  -- Geometric mean: punishes one bad trait hard (very non-linear)
  SET geo = POW((c1+0.001) * (c2+0.001) * (c3+0.001) * (c4+0.001) * (c5+0.001), 1.0/5.0);

  -- Synergy: some trait pairs amplify compatibility
  SET synergy =
      0.6 * POW(c2 * c4, 0.65) +
      0.4 * POW(c1 * c5, 0.65);

  -- Dealbreaker penalty: smooth drop if any trait is very mismatched
  -- center=0.30, steepness=14
  SET dealPenalty = 1.0 - (1.0 / (1.0 + EXP(14.0 * (minC - 0.30))));

  SET raw = (0.70 * geo) + (0.30 * synergy);
  SET raw = raw * (1.0 - 0.50 * dealPenalty);

  -- clamp raw
  IF raw < 0 THEN SET raw = 0; END IF;
  IF raw > 1 THEN SET raw = 1; END IF;

  -- 4) Map to 0..100 with logistic (gives spread + hits highs)
  -- center=0.43, steepness=11
  SET scoreBase = 100.0 / (1.0 + EXP(-11.0 * (raw - 0.43)));

  -- clamp and return FLOAT score (no rounding)
  IF scoreBase < 0 THEN SET scoreBase = 0; END IF;
  IF scoreBase > 100 THEN SET scoreBase = 100; END IF;

  RETURN scoreBase;
END
;;
delimiter ;

SET FOREIGN_KEY_CHECKS = 1;
