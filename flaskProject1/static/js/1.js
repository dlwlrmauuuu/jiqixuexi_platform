  var socket = io();
  const carry = document.getElementById('carry');
  function trans() {
    var division_method_element = document.querySelector('.huafenfangfa select');  // 划分方法下拉列表
    var model_element = document.querySelector('.moxing select');        // 模型下拉列表
    var division_rate_element = document.getElementById('slider1');        // 分割比例滑块



    var evaluation_element = document.querySelectorAll('input[type="checkbox"]');  //评价指标

    var evaluation_element_values = [];   //评价指标的值

      // 遍历每个复选框
    for (var i = 0; i < evaluation_element.length; i++) {
      // 检查复选框是否被选中
      if (evaluation_element[i].checked) {
      // 如果被选中，将其值添加到数组中
      evaluation_element_values.push(evaluation_element[i].value);
      }
    }


    var private_para_element  = document.getElementById("divs");

    var private_para_values = [];
    var private_para_selected = private_para_element.querySelectorAll("input[type='text']");

    for (var k = 0; k < private_para_selected.length; k++) {
        if(check(private_para_selected[k].id,private_para_selected[k].value)){
            alert("参数"+private_para_selected[k].id+"输入存在错误!")
            return false
        }
    var selectedOption = private_para_selected[k].value;
    private_para_values.push(selectedOption);
    }


    var division_method_value = division_method_element.options[division_method_element.selectedIndex].value;
    var model_value = model_element.options[model_element.selectedIndex].value;
    var division_rate_value = division_rate_element.value;


    var data = {
      "shujuji" : 0,
      "huafenfangfa": division_method_value,
      "moxing": model_value,
      "fengebili": division_rate_value,
      "pingjiazhibiao":evaluation_element_values,
      "private_para":private_para_values
    };


    socket.emit('client_message', data);
  }

  carry.addEventListener('click', function () {
    trans();
  });


  socket.on('log_message', function(message) {
    if (message.type === 'print') {
      var logDiv = document.getElementById('log');
      var name = message.name
      var content = message.message;
      logDiv.innerHTML +='['+name +']'+': '+ content + '</br>';
      }
  });
  socket.on('server_message',function (backdata) {
    var outcome_rate = document.getElementById('outcome_1')
    // var content = 'Outcome: ' + backdata.outcome.toString();
    var content2 = ' Evaluate_standard: '+'</br>' + backdata.evaluation_standard;
    outcome_rate.innerHTML += content2;
  });


  function showPlaceholder(str) {
  if (str == "degree" || str == "tol" || str == "k" || str == "learning_rate" || str == "iterations" || str == "max_depth" || str == "d"
  || str == "n_clusters" || str == "max_iter" || str == "n_estimators" || str == "random_state" || str == "n_components" || str == "n_components"
  ||str=="eta"||str=="subsample")
  return "请输入数字";
  else if (str == "kernel" || str == "initCent")
  return "请输入字符";
  else if (str == "kernel")
  return "请输入rbf或linear或poly或sigmoid"
  else if(str=="criterion")
  return "请输入gini或entropy或error"
  else if (str == "whiten")
  return "请输入True或False";
  else if (str == "flag")
  return "请输入0或1";
  else
  return "";
  }

  function showDivs() {
    var selectedOption = document.getElementById("option").value;

  if (selectedOption !== "") {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/get_divs", true);  // 发送POST请求至后端获取新的div和内容
    xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    xhr.onreadystatechange = function() {
      if (xhr.readyState === 4 && xhr.status === 200) {
        var divs = JSON.parse(xhr.responseText);

        var divsContainer = document.getElementById("divs");
        divsContainer.innerHTML = "";  // 清空原有的div

        for (var i = 0; i < divs.length; i++) {
          var div = document.createElement("div");
          var inputLabel = document.createElement("label");
          inputLabel.innerHTML = divs[i];
          var inputTextbox = document.createElement("input");
          inputTextbox.type = "text";
          inputTextbox.id=divs[i];
          inputTextbox.placeholder=showPlaceholder(divs[i])

          inputLabel.appendChild(inputTextbox);
          div.appendChild(inputLabel);
          divsContainer.appendChild(div);
        }
      }
    };
    // 将输入的文本发送给后端
    var data = "option=" + encodeURIComponent(selectedOption);
    xhr.send(data);
  } else {
    document.getElementById("divs").innerHTML = "";  // 清空divs元素
  }
}


  function check(str,value) {
  if (str == "degree" || str == "tol" || str == "k" || str == "learning_rate" || str == "iterations" || str == "max_depth" || str == "d"
  || str == "n_clusters" || str == "max_iter" || str == "n_estimators" || str == "random_state" || str == "n_components" || str == "n_components"
  ||str=="eta"||str=="subsample")
  return isNaN(value);
  else if (str == "kernel" || str == "initCent")
  return !isNaN(value);
  else if (str == "kernel")
  return value!=="rbf"&&value!=="linear"&&value!=="poly"&&value!=="sigmoid";
  else if(str=="criterion")
  return value!=="gini"&&value!=="entropy"&&value!=="error"
  else if (str == "whiten")
  return value!=="True"&&value!=="true"&&value!=="False"&&value!=="false";
  else if (str == "flag")
  return value !== "0" && value !== "1";
  else
  return false;
  }

