import org.apache.spark.sql._
import org.apache.spark.broadcast._

import org.apache.spark.ml.recommendation._
import scala.util.Random

// load user data
val rawUserArtistData = spark.read.textFile("hdfs:///user/jesse/ds/user_artist_data.txt")

// convert to DataFrame
val userArtistDF = rawUserArtistData.map { line =>
	val Array(user, artist, _*) = line.split(' ')
	(user.toInt, artist.toInt)
}.toDF("user", "artist")

// load artist data
val rawArtistData = spark.read.textFile("hdfs:///user/jesse/ds/artist_data.txt")

// convert to DataFrame
val artistByID = rawArtistData.flatMap { line =>
	val (id, name) = line.span(_ != '\t')
	if (name.isEmpty) {
		None
	} else {
		try {
			Some((id.toInt, name.trim))
		} catch {
			case _: NumberFormatException => None
		}
	}
}.toDF("id", "name")

// load artist aliases
val rawArtistAlias = spark.read.textFile("hdfs:///user/jesse/ds/artist_alias.txt")
val artistAlias = rawArtistAlias.flatMap { line =>
	val Array(artist, alias) = line.split('\t') // Declare an array. For each line, split on the tab
	if (artist.isEmpty) { // if the artist is missing
		None
	} else {
		Some((artist.toInt, alias.toInt)) // Return the artist and alias or 0/Null else
	}
}.collect().toMap // store to map

def buildCounts (
		rawUserArtistData: Dataset[String],
		bArtistAlias: Broadcast[Map[Int,Int]]): DataFrame = {
	rawUserArtistData.map { line =>
		val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
		val finalArtistID =
			bArtistAlias.value.getOrElse(artistID, artistID)
		(userID, finalArtistID, count)
	}.toDF("user", "artist", "count")
}

val bArtistAlias = spark.sparkContext.broadcast(artistAlias)

val trainData = buildCounts(rawUserArtistData, bArtistAlias)
trainData.cache()

val model = new ALS().
	setSeed(Random.nextLong()).
	setImplicitPrefs(true).
	setRank(10).
	setRegParam(0.01).
	setAlpha(1.0).
	setMaxIter(5).
	setUserCol("user").
	setItemCol("artist").
	setRatingCol("count").
	setPredictionCol("prediction").
	fit(trainData)