#pragma once

namespace dataloader {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	public ref class DataLoaderForm : public System::Windows::Forms::Form {
	public:
		DataLoaderForm(void) {
			InitializeComponent();
		}

		DataLoaderForm(array<String^>^ args) {
			InitializeComponent();
			args = args;
		}

	protected:
		~DataLoaderForm() {
			if (components)
			{
				delete components;
			}
		}

	private:
		array<String^>^ args;
		AxJVDTLabLib::AxJVLink^ jvlink;
		System::ComponentModel::Container^ components;
		System::Windows::Forms::Button^ button;
		System::Windows::Forms::TextBox^ textBox;

		void InitializeComponent(void) {

			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(DataLoaderForm::typeid));
			this->jvlink = (gcnew AxJVDTLabLib::AxJVLink());
			this->button = (gcnew System::Windows::Forms::Button());
			this->textBox = (gcnew System::Windows::Forms::TextBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->jvlink))->BeginInit();
			this->SuspendLayout();

			this->jvlink->Enabled = true;
			this->jvlink->Location = System::Drawing::Point(13, 13);
			this->jvlink->Name = L"jvlink";
			this->jvlink->OcxState = (cli::safe_cast<System::Windows::Forms::AxHost::State^>(resources->GetObject(L"jvlink.OcxState")));
			this->jvlink->Size = System::Drawing::Size(192, 192);
			this->jvlink->TabIndex = 0;

			this->button->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->button->Location = System::Drawing::Point(322, 211);
			this->button->Name = L"button";
			this->button->Size = System::Drawing::Size(75, 23);
			this->button->TabIndex = 1;
			this->button->Text = L"Exit";
			this->button->UseVisualStyleBackColor = true;
			this->button->Click += gcnew System::EventHandler(this, &DataLoaderForm::button_Click);

			this->textBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
				| System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->textBox->Location = System::Drawing::Point(13, 13);
			this->textBox->Multiline = true;
			this->textBox->Name = L"textBox";
			this->textBox->ScrollBars = System::Windows::Forms::ScrollBars::Both;
			this->textBox->Size = System::Drawing::Size(384, 192);
			this->textBox->TabIndex = 2;

			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(410, 246);
			this->Controls->Add(this->jvlink);
			this->Controls->Add(this->textBox);
			this->Controls->Add(this->button);
			this->Name = L"form";
			this->Text = L"Data Loader";
			this->Load += gcnew System::EventHandler(this, &DataLoaderForm::form_Load);
			this->Shown += gcnew System::EventHandler(this, &DataLoaderForm::form_Shown);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->jvlink))->EndInit();
			this->ResumeLayout(false);
		}

		System::Void form_Load(System::Object^ sender, System::EventArgs^ e) {
		}

		System::Void form_Shown(System::Object^ sender, System::EventArgs^ e) {
			// TODO: コマンドライン引数を処理する

			// params
			String^ sid = "UNKNOWN";
			String^ dataSpec = "RACE";
			String^ fromTime = "20190101000000-20230101000000";
			int option = 4;
			int size = 110000;

			// jvlink variables
			int retVal;
			int readCount;
			int downloadCount;
			String^ lastFileTimeStamp;
			String^ fileName;
			String^ buff;

			// other variables
			String^ statusMessage;
			int previousStatusLength;
			int textLength;
			int progress;
			int lastRetVal;

			// JVInit
			retVal = jvlink->JVInit(sid);
			if (retVal == 0) {
				textBox->AppendText("JVLink initialized successfully.\r\n");
			}
			else {
				textBox->AppendText("JVLink initialization failed with error code: " + retVal.ToString() + "\r\n");
				return;
			}

			// show dataSpec, fromTime, option
			textBox->AppendText("Data Spec: " + dataSpec + "\r\n");
			textBox->AppendText("From Time: " + fromTime + "\r\n");
			textBox->AppendText("Option: " + option.ToString() + "\r\n");

			// JVOpen
			retVal = jvlink->JVOpen(dataSpec, fromTime, option, readCount, downloadCount, lastFileTimeStamp);
			if (retVal == 0) {
				textBox->AppendText("JVLink opened successfully.\r\n");
			}
			else {
				textBox->AppendText("JVLink open failed with error code: " + retVal.ToString() + "\r\n");
				return;
			}

			// show readCount, downloadCount, lastFileTimeStamp
			textBox->AppendText("Read Count: " + readCount.ToString() + "\r\n");
			textBox->AppendText("Download Count: " + downloadCount.ToString() + "\r\n");
			textBox->AppendText("Last File Time Stamp: " + lastFileTimeStamp + "\r\n");

			// JVStatus
			retVal = 0;
			lastRetVal = -1;
			previousStatusLength = 0;
			while (true) {
				if (retVal >= 0) {
					if (lastRetVal != retVal) {
						lastRetVal = retVal;
						// create message
						statusMessage = "Downloading file: " + retVal.ToString() + " of " + downloadCount.ToString() + "\r\n";

						// remove previous status line
						textLength = textBox->Text->Length;
						textBox->Text = textBox->Text->Remove(textLength - previousStatusLength, previousStatusLength);

						// append new status
						textBox->AppendText(statusMessage);
						previousStatusLength = statusMessage->Length;

						if (retVal == downloadCount) {
							// all files downloaded
							textBox->AppendText("All files downloaded.\r\n");
							break;
						}
					}
				}
				else {
					// download failed
					textBox->AppendText("Download failed with error code: " + retVal.ToString() + "\r\n");
					return;
				}
				Application::DoEvents();
				retVal = jvlink->JVStatus();
			}

			// JVRead
			progress = -1;
			retVal = -1;
			previousStatusLength = 0;
			while (true) {
				if (retVal > 0) {
					// write to file
				}
				else if (retVal == -1) {
					++progress;
					// show progress
					statusMessage = "Reading file: " + progress.ToString() + " of " + readCount.ToString() + "\r\n";

					// remove previous status line
					textLength = textBox->Text->Length;
					textBox->Text = textBox->Text->Remove(textLength - previousStatusLength, previousStatusLength);

					// append new status
					textBox->AppendText(statusMessage);
					previousStatusLength = statusMessage->Length;
				}
				else if (retVal == 0) {
					// all files read
					textBox->AppendText("All files read.\r\n");
					break;
				}
				else {
					// read failed
					textBox->AppendText("Read failed with error code: " + retVal.ToString() + "\r\n");
					return;
				}
				Application::DoEvents();
				retVal = jvlink->JVRead(buff, size, fileName);
			}







			// done
			textBox->AppendText("Done.\r\n");

			Application::Exit();
		}

		System::Void button_Click(System::Object^ sender, System::EventArgs^ e) {
			Application::Exit();
		}
	};
}
