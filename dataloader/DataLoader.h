#pragma once
ref class Config;
ref class DataLoader
{
private:
	AxJVDTLabLib::AxJVLink^ jvlink;
	Config^ config;
	bool Initialize();

public:
	DataLoader(AxJVDTLabLib::AxJVLink^);
	bool Execute();
};

