#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;

RecordProcessor::RecordProcessor(NpgsqlConnection^ connection) {
	this->connection = connection;
}

bool RecordProcessor::Initialize() {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS ra ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"kyori SMALLINT, "
			"track_code CHAR(2), "
			"course_kubun CHAR(2), "
			"tenko_code CHAR(1), "
			"babajotai_code CHAR(1), "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS se ("
			"data_type VARCHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"wakuban INT, "
			"umaban INT, "
			"ketto_toroku_bango BIGINT, "
			"barei INT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, "
			"umaban, ketto_toroku_bango))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS um ("
			"data_type VARCHAR(1), "
			"creation_date DATE, "
			"ketto_toroku_bango BIGINT, "
			"birth_date DATE, "
			"seibetsu_code CHAR(1), "
			"hanshoku_toroku_bango_01 BIGINT, "
			"hanshoku_toroku_bango_02 BIGINT, "
			"hanshoku_toroku_bango_03 BIGINT, "
			"hanshoku_toroku_bango_04 BIGINT, "
			"hanshoku_toroku_bango_05 BIGINT, "
			"hanshoku_toroku_bango_06 BIGINT, "
			"hanshoku_toroku_bango_07 BIGINT, "
			"hanshoku_toroku_bango_08 BIGINT, "
			"hanshoku_toroku_bango_09 BIGINT, "
			"hanshoku_toroku_bango_10 BIGINT, "
			"hanshoku_toroku_bango_11 BIGINT, "
			"hanshoku_toroku_bango_12 BIGINT, "
			"hanshoku_toroku_bango_13 BIGINT, "
			"hanshoku_toroku_bango_14 BIGINT, "
			"chokyoshi_code INT, "
			"PRIMARY KEY (ketto_toroku_bango))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS hn ("
			"data_type VARCHAR(1), "
			"creation_date DATE, "
			"hanshoku_toroku_bango BIGINT, "
			"ketto_toroku_bango BIGINT, "
			"PRIMARY KEY (hanshoku_toroku_bango))";
		command->ExecuteNonQuery();

		command->CommandText =
			"ALTER TABLE se ADD CONSTRAINT fk_ra "
			"FOREIGN KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango) "
			"REFERENCES ra(kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango) "
			"DEFERRABLE INITIALLY DEFERRED; "

			"ALTER TABLE se ADD CONSTRAINT fk_um "
			"FOREIGN KEY (ketto_toroku_bango) "
			"REFERENCES um(ketto_toroku_bango) "
			"DEFERRABLE INITIALLY DEFERRED";
		command->ExecuteNonQuery();

		command->CommandText =
			"ALTER TABLE um ADD CONSTRAINT fk_hn_01 "
			"FOREIGN KEY (hanshoku_toroku_bango_01) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_02 "
			"FOREIGN KEY (hanshoku_toroku_bango_02) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_03 "
			"FOREIGN KEY (hanshoku_toroku_bango_03) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_04 "
			"FOREIGN KEY (hanshoku_toroku_bango_04) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_05 "
			"FOREIGN KEY (hanshoku_toroku_bango_05) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_06 "
			"FOREIGN KEY (hanshoku_toroku_bango_06) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_07 "
			"FOREIGN KEY (hanshoku_toroku_bango_07) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_08 "
			"FOREIGN KEY (hanshoku_toroku_bango_08) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_09 "
			"FOREIGN KEY (hanshoku_toroku_bango_09) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_10 "
			"FOREIGN KEY (hanshoku_toroku_bango_10) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_11 "
			"FOREIGN KEY (hanshoku_toroku_bango_11) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_12 "
			"FOREIGN KEY (hanshoku_toroku_bango_12) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_13 "
			"FOREIGN KEY (hanshoku_toroku_bango_13) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED; "
			"ALTER TABLE um ADD CONSTRAINT fk_hn_14 "
			"FOREIGN KEY (hanshoku_toroku_bango_14) REFERENCES hn(hanshoku_toroku_bango) DEFERRABLE INITIALLY DEFERRED";
		command->ExecuteNonQuery();

		command->CommandText =
			"ALTER TABLE hn ADD CONSTRAINT fk_um "
			"FOREIGN KEY (ketto_toroku_bango) REFERENCES um(ketto_toroku_bango) DEFERRABLE INITIALLY DEFERRED";
		command->ExecuteNonQuery();

		return true;
	}
	catch (Exception^ ex) {
		Console::WriteLine("Failed to create tables: {0}", ex->Message);
		return false;
	}
}

// PROCESS_ERROR を返す場合は改行すること
int RecordProcessor::ProcessRecord(String^ record) {
	try {
		String^ recordTypeId = record->Substring(0, 2);
		int result = PROCESS_SKIP;

		if (recordTypeId == "RA")
			result = ProcessRaRecord(record);
		else if (recordTypeId == "SE")
			result = ProcessSeRecord(record);
		else if (recordTypeId == "UM")
			result = ProcessUmRecord(record);

		return result;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing record: {0}", ex->Message);
	}
}

int RecordProcessor::ProcessRaRecord(String^ record) {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		String^ dataType = record->Substring(2, 1);
		DateTime^ creationDate = DateTime::ParseExact(record->Substring(3, 8), "yyyyMMdd", nullptr);
		DateTime^ kaisaiDate = DateTime::ParseExact(record->Substring(11, 8), "yyyyMMdd", nullptr);
		String^ keibajoCode = record->Substring(19, 2);
		int kaisaiKai = Int32::Parse(record->Substring(21, 2));
		int kaisaiNichime = Int32::Parse(record->Substring(23, 2));
		int kyosoBango = Int32::Parse(record->Substring(25, 2));
		int kyori = Int32::Parse(record->Substring(697, 4));
		String^ trackCode = record->Substring(705, 2);
		String^ courseKubun = record->Substring(709, 2);
		String^ tenkoCode = record->Substring(887, 1);
		String^ babajotaiCode;
		int trackCodeInt = Int32::Parse(trackCode);
		if ((23 <= trackCodeInt && trackCodeInt <= 29) || trackCodeInt == 52)
			babajotaiCode = record->Substring(889, 1);
		else if ((10 <= trackCodeInt && trackCodeInt <= 22) || (51 <= trackCodeInt && trackCodeInt <= 59))
			babajotaiCode = record->Substring(888, 1);
		else {
			Console::WriteLine();
			Console::WriteLine("Invalid trackCode: {0}", trackCode);
			return PROCESS_ERROR;
		}

		// 既存のレコードがあるか確認
		command->CommandText = "SELECT creation_date FROM ra WHERE "
			"kaisai_date = @kaisai_date AND "
			"keibajo_code = @keibajo_code AND "
			"kaisai_kai = @kaisai_kai AND "
			"kaisai_nichime = @kaisai_nichime AND "
			"kyoso_bango = @kyoso_bango";
		command->Parameters->Clear();
		command->Parameters->AddWithValue("@kaisai_date", kaisaiDate);
		command->Parameters->AddWithValue("@keibajo_code", keibajoCode);
		command->Parameters->AddWithValue("@kaisai_kai", kaisaiKai);
		command->Parameters->AddWithValue("@kaisai_nichime", kaisaiNichime);
		command->Parameters->AddWithValue("@kyoso_bango", kyosoBango);
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// 既存のレコードがない場合は挿入して終了
		if (existingCreationDateObj == nullptr) {
			command->CommandText = "INSERT INTO ra (data_type, creation_date, "
				"kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, "
				"kyori, track_code, course_kubun, tenko_code, babajotai_code) "
				"VALUES (@data_type, @creation_date, "
				"@kaisai_date, @keibajo_code, @kaisai_kai, @kaisai_nichime, @kyoso_bango, "
				"@kyori, @track_code, @course_kubun, @tenko_code, @babajotai_code)";
			command->Parameters->Clear();
			command->Parameters->AddWithValue("@data_type", dataType);
			command->Parameters->AddWithValue("@creation_date", creationDate);
			command->Parameters->AddWithValue("@kaisai_date", kaisaiDate);
			command->Parameters->AddWithValue("@keibajo_code", keibajoCode);
			command->Parameters->AddWithValue("@kaisai_kai", kaisaiKai);
			command->Parameters->AddWithValue("@kaisai_nichime", kaisaiNichime);
			command->Parameters->AddWithValue("@kyoso_bango", kyosoBango);
			command->Parameters->AddWithValue("@kyori", kyori);
			command->Parameters->AddWithValue("@track_code", trackCode);
			command->Parameters->AddWithValue("@course_kubun", courseKubun);
			command->Parameters->AddWithValue("@tenko_code", tenkoCode);
			command->Parameters->AddWithValue("@babajotai_code", babajotaiCode);
			command->ExecuteNonQuery();
			return PROCESS_SUCCESS;
		}

		// 既存のレコードがある場合は日付を確認
		DateTime^ existingCreationDate = Convert::ToDateTime(existingCreationDateObj);

		// 既存の日付の方が新しい場合は何もしないで終了
		if (existingCreationDate->CompareTo(creationDate) > 0)
			return PROCESS_SUCCESS;

		// 既存の日付が古いか同じ場合は更新して終了
		command->CommandText = "UPDATE ra SET "
			"data_type = @data_type, "
			"creation_date = @creation_date, "
			"kyori = @kyori, "
			"track_code = @track_code, "
			"course_kubun = @course_kubun, "
			"tenko_code = @tenko_code, "
			"babajotai_code = @babajotai_code "
			"WHERE kaisai_date = @kaisai_date "
			"AND keibajo_code = @keibajo_code "
			"AND kaisai_kai = @kaisai_kai "
			"AND kaisai_nichime = @kaisai_nichime "
			"AND kyoso_bango = @kyoso_bango";
		command->Parameters->Clear();
		command->Parameters->AddWithValue("@data_type", dataType);
		command->Parameters->AddWithValue("@creation_date", creationDate);
		command->Parameters->AddWithValue("@kyori", kyori);
		command->Parameters->AddWithValue("@track_code", trackCode);
		command->Parameters->AddWithValue("@course_kubun", courseKubun);
		command->Parameters->AddWithValue("@tenko_code", tenkoCode);
		command->Parameters->AddWithValue("@babajotai_code", babajotaiCode);
		command->Parameters->AddWithValue("@kaisai_date", kaisaiDate);
		command->Parameters->AddWithValue("@keibajo_code", keibajoCode);
		command->Parameters->AddWithValue("@kaisai_kai", kaisaiKai);
		command->Parameters->AddWithValue("@kaisai_nichime", kaisaiNichime);
		command->Parameters->AddWithValue("@kyoso_bango", kyosoBango);
		command->ExecuteNonQuery();
		return PROCESS_SUCCESS;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing RA record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}

int RecordProcessor::ProcessSeRecord(String^ record) {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);
		NpgsqlDataReader^ reader;

		String^ dataType = record->Substring(2, 1);
		DateTime^ creationDate = DateTime::ParseExact(record->Substring(3, 8), "yyyyMMdd", nullptr);
		DateTime^ kaisaiDate = DateTime::ParseExact(record->Substring(11, 8), "yyyyMMdd", nullptr);
		String^ keibajoCode = record->Substring(19, 2);
		int kaisaiKai = Int32::Parse(record->Substring(21, 2));
		int kaisaiNichime = Int32::Parse(record->Substring(23, 2));
		int kyosoBango = Int32::Parse(record->Substring(25, 2));
		int wakuban = Int32::Parse(record->Substring(27, 1));
		int umaban = Int32::Parse(record->Substring(28, 2));
		int kettoTorokuBango = Int64::Parse(record->Substring(30, 10));
		int barei = Int32::Parse(record->Substring(82, 2));

		// 既存のレコードがあるか確認
		command->CommandText =
			"SELECT creation_date FROM se"
			" WHERE kaisai_date = @kaisai_date"
			" AND keibajo_code = @keibajo_code"
			" AND kaisai_kai = @kaisai_kai"
			" AND kaisai_nichime = @kaisai_nichime"
			" AND kyoso_bango = @kyoso_bango"
			" AND umaban = @umaban"
			" AND ketto_toroku_bango = @ketto_toroku_bango";
		command->Parameters->Clear();
		command->Parameters->AddWithValue("@kaisai_date", kaisaiDate);
		command->Parameters->AddWithValue("@keibajo_code", keibajoCode);
		command->Parameters->AddWithValue("@kaisai_kai", kaisaiKai);
		command->Parameters->AddWithValue("@kaisai_nichime", kaisaiNichime);
		command->Parameters->AddWithValue("@kyoso_bango", kyosoBango);
		command->Parameters->AddWithValue("@umaban", umaban);
		command->Parameters->AddWithValue("@ketto_toroku_bango", kettoTorokuBango);
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// 既存のレコードがない場合は挿入して終了
		if (existingCreationDateObj == nullptr) {
			command->CommandText =
				"INSERT INTO se (data_type, creation_date, "
				"kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, "
				"wakuban, umaban, ketto_toroku_bango, "
				"barei"
				") VALUES (@data_type, @creation_date, "
				"@kaisai_date, @keibajo_code, @kaisai_kai, @kaisai_nichime, @kyoso_bango, "
				"@wakuban, @umaban, @ketto_toroku_bango, "
				"@barei"
				")";
			command->Parameters->Clear();
			command->Parameters->AddWithValue("@data_type", dataType);
			command->Parameters->AddWithValue("@creation_date", creationDate);
			command->Parameters->AddWithValue("@kaisai_date", kaisaiDate);
			command->Parameters->AddWithValue("@keibajo_code", keibajoCode);
			command->Parameters->AddWithValue("@kaisai_kai", kaisaiKai);
			command->Parameters->AddWithValue("@kaisai_nichime", kaisaiNichime);
			command->Parameters->AddWithValue("@kyoso_bango", kyosoBango);
			command->Parameters->AddWithValue("@wakuban", wakuban);
			command->Parameters->AddWithValue("@umaban", umaban);
			command->Parameters->AddWithValue("@ketto_toroku_bango", kettoTorokuBango);
			command->Parameters->AddWithValue("@barei", barei);
			command->ExecuteNonQuery();
			return PROCESS_SUCCESS;
		}

		// 既存のレコードがある場合は日付を確認
		DateTime^ existingCreationDate = Convert::ToDateTime(existingCreationDateObj);

		// 既存の日付の方が新しい場合は何もしないで終了
		if (existingCreationDate->CompareTo(creationDate) > 0)
			return PROCESS_SUCCESS;

		// 既存の日付が古いか同じ場合は更新して終了
		command->CommandText =
			"UPDATE se SET "
			"data_type = @data_type, "
			"creation_date = @creation_date, "
			"wakuban = @wakuban, "
			"barei = @barei "
			"WHERE kaisai_date = @kaisai_date "
			"AND keibajo_code = @keibajo_code "
			"AND kaisai_kai = @kaisai_kai "
			"AND kaisai_nichime = @kaisai_nichime "
			"AND kyoso_bango = @kyoso_bango "
			"AND umaban = @umaban "
			"AND ketto_toroku_bango = @ketto_toroku_bango";
		command->Parameters->Clear();
		command->Parameters->AddWithValue("@data_type", dataType);
		command->Parameters->AddWithValue("@creation_date", creationDate);
		command->Parameters->AddWithValue("@wakuban", wakuban);
		command->Parameters->AddWithValue("@barei", barei);
		command->Parameters->AddWithValue("@kaisai_date", kaisaiDate);
		command->Parameters->AddWithValue("@keibajo_code", keibajoCode);
		command->Parameters->AddWithValue("@kaisai_kai", kaisaiKai);
		command->Parameters->AddWithValue("@kaisai_nichime", kaisaiNichime);
		command->Parameters->AddWithValue("@kyoso_bango", kyosoBango);
		command->Parameters->AddWithValue("@umaban", umaban);
		command->Parameters->AddWithValue("@ketto_toroku_bango", kettoTorokuBango);
		command->ExecuteNonQuery();
		return PROCESS_SUCCESS;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing RA record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}

int RecordProcessor::ProcessUmRecord(String^ record) {
	return PROCESS_SUCCESS;
}
