$(document).ready(function() {
		/*开始时将树的节点都折叠起来*/
		$("#tree ul").hide();
		/*隐藏没有叶子节点的分支*/
		$("#tree .root").each(function() {
			if($(this).next("ul").find(".leaf").length == 0){
				$(this).remove();
			}
		});
	    $("#tree .folder").each(function() {
			if($(this).next("ul").find(".leaf").length == 0){
				$(this).remove();
			}
		});
		/*展开*/
		$("#tree .root").click(function(e) {
		 if($(e.target).get(0).tagName.toLowerCase() == "a"){
			var p = $(e.target).parent();
			p.toggleClass("rootOpen");
			p.next("ul").toggle();
			return;
		 }
		 $(e.target).toggleClass("rootOpen");
		 $(e.target).next("ul").toggle();
		});
		$("#tree .folder").click(function(e) {
		 if($(e.target).get(0).tagName.toLowerCase() == "a"){
		    var p = $(e.target).parent();
		    p.toggleClass("folderOpen");
		    p.next("ul").toggle();
		    return;
		 }
		 $(e.target).toggleClass("folderOpen");
		 $(e.target).next("ul").toggle();
		});
		/*树结构搜索*/
		$("#btn").click(function() {
			var txt = $("#tfd").val();
			if (txt.length == 0) {
				return;
			}
			$("#tree").find(".rootOpen").each(function() {
				$(this).removeClass("rootOpen");
				$(this).next("ul").hide();
			});
			$("#tree").find(".folderOpen").each(function() {
				$(this).removeClass("folderOpen");
				$(this).next("ul").hide();
			});
			$("#tree").find(".leaf a").each(function() {
				$.windowbox.highlightElement($(this), txt);
				if ($(this).text().match(txt) != null) {
					$(this).parents("ul").each(function() {
						$(this).show();
						var li = $(this).prev("li");
						if (li.hasClass("root")) {
							li.addClass("rootOpen");
						} else if (li.hasClass("folder")) {
							li.addClass("folderOpen");
						}
					});
				}
			});
		});
		/*按表名搜索*/
		$("#btn1").click(function() {
			var txt = $("#tfd1").val();
			if (txt.length == 0) {
				return;
			}
			$("#currentResults").html("");
			$("#results").find("ul").each(function() {
				var p = $(this);
				$(this).find("li span").each(function() {
					if ($(this).text().match(txt) != null) {
						$("#currentResults").append(p.clone());
						return false;
					}
				});
			});
			$.windowbox.highlightRange("#currentResults ul li span", txt);
		});
		/*按列名搜索*/
		$("#btn2").click(function() {
			var txt = $("#tfd2").val();
			if (txt.length == 0) {
				return;
			}
			$("#currentResults1").html("");
			$("#results1").find("ul").each(function() {
				var p = $(this);
				$(this).find("li span").each(function() {
					if ($(this).text().match(txt) != null) {
						$("#currentResults1").append(p.clone());
						return false;
					}
				});
			});
			$.windowbox.highlightRange("#currentResults1 ul li span", txt);
		});

		$.windowbox = {
			/*高亮标签*/
			highlightRange : function(range, key) {
				$(range).each(function() {
					$(this).html($(this).text());
					$(this).html($(this).html().replace(new RegExp(key, 'g'), "<span class=\"highlight\">" + key + "</span>"));
				});
			},
			/*高亮某个元素*/
			highlightElement : function(element, key) {
				element.html(element.text());
				element.html(element.html().replace(new RegExp(key, 'g'), "<span class=\"highlight\">" + key + "</span>"));
			}
		}
	});