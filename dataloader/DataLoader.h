#pragma once
ref class Config;
ref class DataLoader {
private:
	ref struct JVOpenParams {
		System::String^ dataSpec;
		System::String^ fromDate;
		int option;
		JVOpenParams(System::String^ dataSpec, System::String^ fromDate, int option) {
			this->dataSpec = dataSpec;
			this->fromDate = fromDate;
			this->option = option;
		}
	};
	AxJVDTLabLib::AxJVLink^ jvlink;
	Config^ config;
	bool Initialize();
	bool ProcessChunk(JVOpenParams^);
	int WaitForDownloadCompletion(int downloadCount);

public:
	DataLoader(AxJVDTLabLib::AxJVLink^);
	bool Execute();
};

