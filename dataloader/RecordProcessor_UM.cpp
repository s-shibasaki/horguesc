#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;



int RecordProcessor::ProcessUMRecord(String^ record) {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		command->Parameters->AddWithValue("@data_type", record->Substring(2, 1));

		DateTime^ creationDate = DateTime::ParseExact(record->Substring(3, 8), "yyyyMMdd", nullptr);
		command->Parameters->AddWithValue("@creation_date", creationDate);

		command->Parameters->AddWithValue("@ketto_toroku_bango", Int64::Parse(record->Substring(11, 10)));
		command->Parameters->AddWithValue("@birth_date", DateTime::ParseExact(record->Substring(38, 8), "yyyyMMdd", nullptr));
		command->Parameters->AddWithValue("@seibetsu_code", record->Substring(200, 1));
		for (int i = 0; i < 14; i++)
			command->Parameters->AddWithValue(String::Format("@hanshoku_toroku_bango_{0:00}", i + 1), Int64::Parse(record->Substring(204 + (46 * i), 10)));
		command->Parameters->AddWithValue("@chokyoshi_code", Int32::Parse(record->Substring(849, 5)));

		// �����̃��R�[�h�����邩�m�F
		command->CommandText = "SELECT creation_date FROM um WHERE ketto_toroku_bango = @ketto_toroku_bango";
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// �����̃��R�[�h���Ȃ��ꍇ�͑}�����ďI��
		if (existingCreationDateObj == nullptr) {
			command->CommandText = 
				"INSERT INTO um (data_type, creation_date, "
				"ketto_toroku_bango, birth_date, seibetsu_code, ";
			for (int i = 1; i <= 14; i++)
				command->CommandText += String::Format("hanshoku_toroku_bango_{0:00}, ", i);
			command->CommandText +=
				"chokyoshi_code"
				") VALUES (@data_type, @creation_date, "
				"@ketto_toroku_bango, @birth_date, @seibetsu_code, ";
			for (int i = 1; i <= 14; i++)
				command->CommandText += String::Format("@hanshoku_toroku_bango_{0:00}, ", i);
			command->CommandText +=
				"@chokyoshi_code)";
			command->ExecuteNonQuery();
			return PROCESS_SUCCESS;
		}

		// �����̃��R�[�h������ꍇ�͓��t���m�F
		DateTime^ existingCreationDate = Convert::ToDateTime(existingCreationDateObj);

		// �����̓��t�̕����V�����ꍇ�͉������Ȃ��ŏI��
		if (existingCreationDate->CompareTo(creationDate) > 0)
			return PROCESS_SUCCESS;

		// �����̓��t���Â��������ꍇ�͍X�V���ďI��
		command->CommandText =
			"UPDATE um SET "
			"data_type = @data_type, "
			"creation_date = @creation_date, "
			"birth_date = @birth_date, "
			"seibetsu_code = @seibetsu_code, ";
		for (int i = 1; i <= 14; i++)
			command->CommandText += String::Format("hanshoku_toroku_bango_{0:00} = @hanshoku_toroku_bango_{0:00}, ", i);
		command->CommandText +=
			"chokyoshi_code = @chokyoshi_code "
			"WHERE ketto_toroku_bango = @ketto_toroku_bango";
		command->ExecuteNonQuery();
		return PROCESS_SUCCESS;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing UM record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}