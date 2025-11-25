
using System.Xml;

        XmlDocument doc = new XmlDocument();
        
        string tentep = @"D:\CaoThuHue\Bai3\nhanvien.xml";
        
        int d;
        
        private void HienThi()
        
        {
        
            datanhanvien.Rows.Clear();
            
            doc.Load(tentep);
            
            XmlNodeList DS = doc.SelectNodes("/ds/nhanvien");
            
            int sd = 0;
            
            datanhanvien.ColumnCount = 4;
            
            datanhanvien.Rows.Add();

            foreach (XmlNode nhan_vien in DS)
            {
                XmlNode ma_nv = nhan_vien.SelectSingleNode("@manv");
                datanhanvien.Rows[sd].Cells[0].Value = ma_nv.InnerText.ToString();
                XmlNode ho = nhan_vien.SelectSingleNode("hoten/ho");
                datanhanvien.Rows[sd].Cells[1].Value = ho.InnerText.ToString();
                XmlNode ten = nhan_vien.SelectSingleNode("hoten/ten");
                datanhanvien.Rows[sd].Cells[2].Value = ten.InnerText.ToString();
                XmlNode dia_chi = nhan_vien.SelectSingleNode("diachi");
                datanhanvien.Rows[sd].Cells[3].Value = dia_chi.InnerText.ToString();
                datanhanvien.Rows.Add();
                sd++;
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            HienThi();
        }

        private void bt_them_Click(object sender, EventArgs e)
        {
            doc.Load(tentep);
            XmlElement goc = doc.DocumentElement;

            XmlNode nhan_vien = doc.CreateElement("nhanvien");
            XmlAttribute ma_nv = doc.CreateAttribute("manv");
            ma_nv.InnerText = txt_ma.Text;
            nhan_vien.Attributes.Append(ma_nv);

            XmlNode ho_ten = doc.CreateElement("hoten");
            XmlNode ho = doc.CreateElement("ho");
            ho.InnerText = txt_ho.Text;
            ho_ten.AppendChild(ho);

            XmlNode ten = doc.CreateElement("ten");
            ten.InnerText = txt_ten.Text;
            ho_ten.AppendChild(ten);

            nhan_vien.AppendChild(ho_ten);

            XmlNode dia_chi = doc.CreateElement("diachi");
            dia_chi.InnerText = txt_diachi.Text;
            nhan_vien.AppendChild (dia_chi);

            goc.AppendChild(nhan_vien);

            doc.Save(tentep);
            HienThi();

        }

        private void bt_sua_Click(object sender, EventArgs e)
        {
            doc.Load(tentep);
            XmlElement goc = doc.DocumentElement;

            XmlNode nhanviencu = goc.SelectSingleNode("/ds/nhanvien[@manv='" + txt_ma.Text + "']");
            XmlNode nhanvienmoi = doc.CreateElement("nhanvien");

            XmlAttribute ma_nv = doc.CreateAttribute("manv");
            ma_nv.InnerText = txt_ma.Text;
            nhanvienmoi.Attributes.Append(ma_nv);

            XmlNode ho_ten = doc.CreateElement("hoten");
            XmlNode ho = doc.CreateElement("ho");
            ho.InnerText = txt_ho.Text;
            ho_ten.AppendChild(ho);

            XmlNode ten = doc.CreateElement("ten");
            ten.InnerText = txt_ten.Text;
            ho_ten.AppendChild(ten);

            nhanvienmoi.AppendChild(ho_ten);

            XmlNode dia_chi = doc.CreateElement("diachi");
            dia_chi.InnerText = txt_diachi.Text;
            nhanvienmoi.AppendChild(dia_chi);

            goc.ReplaceChild(nhanvienmoi, nhanviencu);
            doc.Save(tentep);
            HienThi();

        }

        private void bt_xoa_Click(object sender, EventArgs e)
        {
            doc.Load(tentep);
            XmlElement goc = doc.DocumentElement;

            XmlNode nhan_vien_xoa = goc.SelectSingleNode("/ds/nhanvien[@manv='" + txt_ma.Text + "']");
            goc.RemoveChild(nhan_vien_xoa);

            doc.Save(tentep);
            HienThi() ;
        }

        private void bt_tim_Click(object sender, EventArgs e)
        {
            
            doc.Load(tentep);
            XmlElement goc = doc.DocumentElement;
            datanhanvien.Rows.Clear();
            XmlNode nhan_vien_tim = goc.SelectSingleNode("/ds/nhanvien[@manv='" + txt_ma.Text + "']");
            int sd = 0;
            if (nhan_vien_tim != null)
            {

                XmlNode ma_nv = nhan_vien_tim.SelectSingleNode("@manv");
                datanhanvien.Rows[sd].Cells[0].Value = ma_nv.InnerText.ToString();
                XmlNode ho = nhan_vien_tim.SelectSingleNode("hoten/ho");
                datanhanvien.Rows[sd].Cells[1].Value = ho.InnerText.ToString();
                XmlNode ten = nhan_vien_tim.SelectSingleNode("hoten/ten");
                datanhanvien.Rows[sd].Cells[2].Value = ten.InnerText.ToString();
                XmlNode dia_chi = nhan_vien_tim.SelectSingleNode("diachi");
                datanhanvien.Rows[sd].Cells[3].Value = dia_chi.InnerText.ToString();
    
        
            }
        }

        private void datanhanvien_CellClick(object sender, DataGridViewCellEventArgs e)
        {
            d = e.RowIndex;
            txt_ma.Text = datanhanvien.Rows[d].Cells[0].Value.ToString();
            txt_ho.Text = datanhanvien.Rows[d].Cells[1].Value.ToString();
            txt_ten.Text = datanhanvien.Rows[d].Cells[2].Value.ToString();
            txt_diachi.Text = datanhanvien.Rows[d].Cells[3].Value.ToString();
        }
    }
}
