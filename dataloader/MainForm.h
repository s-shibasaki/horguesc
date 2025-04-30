﻿#pragma once

#include "DataLoader.h"

namespace dataloader {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// MainForm の概要
	/// </summary>
	public ref class MainForm : public System::Windows::Forms::Form
	{
	public:
		MainForm(array<String^>^ args)
		{
			this->args = args;
			InitializeComponent();
			//
			//TODO: ここにコンストラクター コードを追加します
			//
		}

	protected:
		/// <summary>
		/// 使用中のリソースをすべてクリーンアップします。
		/// </summary>
		~MainForm()
		{
			if (components)
			{
				delete components;
			}
		}

	private: 
		AxJVDTLabLib::AxJVLink^ jvlink;
		array<String^>^ args;

		/// <summary>
		/// 必要なデザイナー変数です。
		/// </summary>
		System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// デザイナー サポートに必要なメソッドです。このメソッドの内容を
		/// コード エディターで変更しないでください。
		/// </summary>
		void InitializeComponent(void)
		{
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(MainForm::typeid));
			this->jvlink = (gcnew AxJVDTLabLib::AxJVLink());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->jvlink))->BeginInit();
			this->SuspendLayout();
			// 
			// jvlink
			// 
			this->jvlink->Enabled = true;
			this->jvlink->Location = System::Drawing::Point(13, 13);
			this->jvlink->Name = L"jvlink";
			this->jvlink->OcxState = (cli::safe_cast<System::Windows::Forms::AxHost::State^>(resources->GetObject(L"jvlink.OcxState")));
			this->jvlink->Size = System::Drawing::Size(192, 192);
			this->jvlink->TabIndex = 0;
			// 
			// MainForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(284, 261);
			this->Controls->Add(this->jvlink);
			this->Name = L"MainForm";
			this->Text = L"DataLoader";
			this->Load += gcnew System::EventHandler(this, &MainForm::MainForm_Load);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->jvlink))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion

	private:
		System::Void MainForm_Load(System::Object^ sender, System::EventArgs^ e) {
			DataLoader^ dataLoader = gcnew DataLoader(jvlink, args);
			if (!dataLoader->Execute())
				Environment::Exit(1);
			Application::Exit();
		}
	};
}
