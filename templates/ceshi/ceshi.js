$(function() {
   var startX, startY, endX, endY;
   var activeComponent;

   // 绑定组件拖拽事件
   $(".component").mousedown(function(event) {
      event.preventDefault();
      activeComponent = $(this).addClass("active");
      startX = event.pageX - activeComponent.offset().left;
      startY = event.pageY - activeComponent.offset().top;
   });
   $("#workflow-editor").mousemove(function(event) {
      if (activeComponent) {
         var x = event.pageX - $(this).offset().left;
         var y = event.pageY - $(this).offset().top;
         $(".flowlines line.active").attr({
            x2: x,
            y2: y
         });
      }
   });
   $("#workflow-editor").mouseup(function(event) {
      if (activeComponent) {
         var componentId = activeComponent.attr("data-component-id");
         var x = event.pageX - $(this).offset().left;
         var y = event.pageY - $(this).offset().top;
         endX = x;
         endY = y;
         var line = $("<line>").attr({
            class: "active",
            x1: startX,
            y1: startY,
            x2: endX,
            y2: endY
         });
         $(".flowlines").append(line);
         activeComponent.removeClass("active");
         activeComponent = null;
      }
   });

   // 绑定组件双击事件
   $(".component").dblclick(function(event) {
      event.stopPropagation();
      var componentType = $(this).attr("data-component-type");
      var config = $(this).attr("data-component-config");
      var dialog = $("<div>").attr("title", componentType + " Configuration");
      var form = $("<form>");
      dialog.append(form);
      switch (componentType) {
         case "data-source":
            form.append("<p><label>File:</label> <input type='file' value='" + config.file + "'></p>");
            break;
         case "preprocessing":
            form.append("<p><label>Method:</label> <select>" +
            "<option value='method1'>Method 1</option>" +
            "<option value='method2'>Method 2</option>" +
            "<option value='method3'>Method 3</option>" +
            "</select></p>");
            break;
         case "feature-extraction":
            form.append("<p><label>Features:</label> <select multiple>" +
            "<option value='feature1'>Feature 1</option>" +
            "<option value='feature2'>Feature 2</option>" +
            "<option value='feature3'>Feature 3</option>" +
            "</select></p>");
            break;
         case "model-training":
            form.append("<p><label>Algorithm:</label> <select>" +
            "<option value='alg1'>Algorithm 1</option>" +
            "<option value='alg2'>Algorithm 2</option>" +
            "<option value='alg3'>Algorithm 3</option>" +
            "</select></p>");
            break;
         case "model-evaluation":
            form.append("<p><label>Metric:</label> <select>" +
            "<option value='metric1'>Metric 1</option>" +
            "<option value='metric2'>Metric 2</option>" +
            "<option value='metric3'>Metric 3</option>" +
            "</select></p>");
            break; }
dialog.dialog({
resizable: false,
height: "auto",
width: 400,
modal: true,
buttons: {
"Save": function() {
var newConfig = {};
switch (componentType) {
case "data-source":
newConfig.file = dialog.find("input[type='file']").val();
break;
case "preprocessing":
newConfig.method = dialog.find("select").val();
break;
case "feature-extraction":
newConfig.features = dialog.find("select").val();
break;
case "model-training":
newConfig.algorithm = dialog.find("select").val();
break;
case "model-evaluation":
newConfig.metric = dialog.find("select").val();
break;
}
$(this).dialog("close");
$(activeComponent).attr("data-component-config", JSON.stringify(newConfig));
},
"Cancel": function() {
$(this).dialog("close");
}
}
});
});


// 绑定保存按钮事件
$("#save").click(function() {
var workflow = [];
$(".component").each(function() {
var id = $(this).attr("data-component-id");
var type = (this).attr("data-component-type");
         var config = JSON.parse((this).attr("data−component−type"));
         var config = JSON.parse((this).attr("data-component-config"));
var position = {
x: $(this).position().left,
y: $(this).position().top
};
workflow.push({
id: id,
type: type,
config: config,
position: position
});
});
localStorage.setItem("workflow", JSON.stringify(workflow));
});

// 绑定加载按钮事件
$("#load").click(function() {
var workflow = JSON.parse(localStorage.getItem("workflow")) || [];
$(".component").remove();
$(".flowlines line").remove();
$.each(workflow, function(index, component) {
var div = $("<div>").addClass("component").attr({
"data-component-id": component.id,
"data-component-type": component.type,
"data-component-config": JSON.stringify(component.config)
}).text(component.type);
div.css({
left: component.position.x,
top: component.position.y
});
$("#workflow-editor").append(div);
});
});
});
