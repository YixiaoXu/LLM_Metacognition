suppressPackageStartupMessages({library(ggplot2); library(patchwork); library(svglite); library(jsonlite)})
args <- commandArgs(trailingOnly=TRUE)
release <- normalizePath(args[1])
data_dir <- file.path(release, "source_data")
out <- Sys.getenv("METACOG_FIGURE_OUTPUT", file.path(getwd(),"outputs/figures"))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
figdir <- file.path(out,"full_figures")
dir.create(figdir,recursive=TRUE,showWarnings=FALSE)
read <- function(n) read.csv(file.path(data_dir,paste0(n,".csv")),check.names=FALSE)
domains <- c("ultrachat","beavertails","mathqa")
models <- c("llama2_7b","llama32_3b","llama31_8b","qwen3_4b","qwen25_7b","qwen3_8b","deepseek_llama8b","deepseek_qwen7b")
model_names <- c("Llama2-7B","Llama3.2-3B","Llama3.1-8B","Qwen3-4B","Qwen2.5-7B","Qwen3-8B","DS-Llama-8B","DS-Qwen-7B")
domain_names <- c("UltraChat","BeaverTails","MathQA")
blue <- "#0072B2"; light <- "#56B4E9"; warm <- "#D55E00"; grey <- "#A6ADB5"
pal <- c(ultrachat=blue,beavertails=warm,mathqa="#009E73")
domain_colours <- function() scale_colour_manual(values=pal,breaks=domains,labels=domain_names)
theme_set(theme_classic(base_size=7,base_family="Arial") + theme(
  axis.line=element_line(linewidth=.3,colour="#343A43"),axis.ticks=element_line(linewidth=.3),
  axis.text=element_text(colour="#343A43",size=6),axis.title=element_text(size=7),
  strip.background=element_blank(),strip.text=element_text(size=6,face="bold"),
  legend.title=element_text(size=6),legend.text=element_text(size=6),
  legend.key.size=unit(3,"mm"),plot.margin=margin(6,5,4,7),
  legend.margin=margin(0,0,0,0),legend.box.spacing=unit(1,"mm")))
bool <- function(x) tolower(as.character(x)) == "true"
prep <- function(d) {d$dataset <- factor(d$dataset,levels=domains);d$target<-factor(d$target,levels=models);d}
m <- prep(read("modules")); cst <- prep(read("constructs")); nt<-prep(read("next_token"))
eligible_keys <- with(m[bool(m$behavior_pass),],paste(dataset,target,module_key))
nt <- nt[with(nt,paste(dataset,target,module_key)) %in% eligible_keys,]
tr<-prep(read("trajectory")); rc<-prep(read("report_conditions")); rn<-prep(read("report_neurons"))
cond<-prep(read("conditions"))
selected_figure <- if(length(args)>1) as.integer(args[2]) else NA_integer_
layout_manifest <- if(!is.na(selected_figure)) fromJSON(file.path(release,"figure_data/layout.json"),simplifyVector=FALSE) else list()
save_panel <- function(p,id,w,h) {
  svglite::svglite(file.path(out,paste0(id,".svg")),width=w/25.4,height=h/25.4,bg="white")
  print(p);dev.off()
  ggsave(file.path(out,paste0(id,".pdf")),p,width=w,height=h,units="mm",device=cairo_pdf,bg="white")
  if(!is.na(selected_figure) && selected_figure %in% c(2,3,4,5,6)) {
    ggsave(file.path(out,paste0(id,".png")),p,width=w,height=h,units="mm",dpi=300,device="png",type="cairo",bg="white")
  }
}
full_range_inset <- function(main,full,left=.64,bottom=.60,right=.98,top=.97) {
  full <- full + labs(x=NULL,y=NULL,title="Full range") + theme(
    legend.position="none",axis.title=element_blank(),
    axis.text=element_text(size=5),plot.title=element_text(size=5,hjust=.5),
    plot.margin=margin(2,2,2,2),plot.background=element_rect(fill="white",colour="#CED4DA",linewidth=.25))
  # Reserve a separate inset lane: no opaque inset may hide an observation.
  main + (plot_spacer() + inset_element(full,left=0,bottom=.55,right=1,top=.98,align_to="full")) +
    plot_layout(widths=c(2.4,1))
}
captioned <- function(p,label,title) p + labs(caption=paste0(label,"   ",title)) + theme(
  plot.caption=element_text(size=7,face="bold",hjust=.5,margin=margin(t=5)))
export_fig <- function(num,panels,titles,positions,height,file) {
  if(!is.na(selected_figure) && num!=selected_figure) return(invisible(NULL))
  for (i in seq_along(panels)) {
    pos<-positions[[i]]
    save_panel(panels[[i]],paste0("fig",num,"_",letters[i]),pos[3],pos[4])
  }
  # Absolute placement keeps panel boundaries identical in PDF and PowerPoint.
  grDevices::cairo_pdf(file.path(figdir,paste0(file,".pdf")),width=183/25.4,height=height/25.4,family="Arial")
  grid::grid.newpage()
  draw_all<-function() {
    for(i in seq_along(panels)) {
      z<-positions[[i]]
      print(panels[[i]],vp=grid::viewport(x=z[1]/183,y=1-z[2]/height,width=z[3]/183,height=z[4]/height,just=c("left","top")))
      grid::grid.text(letters[i],x=(z[1]+.7)/183,y=1-(z[2]-.8)/height,just=c("left","top"),gp=grid::gpar(fontfamily="Arial",fontsize=8,fontface="bold"))
      grid::grid.text(titles[i],x=(z[1]+z[3]/2)/183,y=1-(z[2]+z[4]+3)/height,gp=grid::gpar(fontfamily="Arial",fontsize=7,fontface="bold"))
    }
  }
  draw_all();dev.off()
  png(file.path(out,paste0("fig",num,"_preview.png")),width=183,height=height,units="mm",res=300,type="cairo")
  grid::grid.newpage();draw_all();dev.off()
  layout_manifest[[as.character(num)]] <<- list(width=183,height=height,file=file,
     panels=lapply(seq_along(panels),function(i) list(letter=letters[i],title=titles[i],file=paste0("fig",num,"_",letters[i],".svg"),position=positions[[i]])))
}
domain_scale <- scale_x_discrete(labels=setNames(domain_names,domains))
distribution <- function(y,label) ggplot(m,aes(x=dataset,y=.data[[y]],fill=dataset,colour=dataset))+
  geom_boxplot(width=.45,outlier.shape=NA,linewidth=.35,alpha=.32)+
  geom_point(position=position_jitter(width=.15,seed=42),alpha=.6,size=.6)+
  scale_fill_manual(values=pal,guide="none")+domain_colours()+guides(colour="none")+domain_scale+labs(x=NULL,y=label)
violin_distribution <- function(y,label) ggplot(m,aes(x=dataset,y=.data[[y]],fill=dataset,colour=dataset))+
  geom_violin(width=.8,scale="width",trim=TRUE,linewidth=.4,alpha=.25)+
  geom_point(position=position_jitter(width=.12,seed=42),alpha=.5,size=.6)+
  geom_boxplot(width=.12,outlier.shape=NA,linewidth=.4,fill="white",alpha=.9)+
  scale_fill_manual(values=pal,guide="none")+domain_colours()+guides(colour="none")+domain_scale+labs(x=NULL,y=label)
fig2_y_range<-c(0,.4)
fig2_y_breaks<-seq(0,.4,by=.1)
fig2_mi_range<-c(0,.10)
fig2_mi_breaks<-seq(0,.10,by=.02)
semantic_reference_region <- function() list(
  annotate("rect",xmin=0,xmax=.033,ymin=-Inf,ymax=Inf,fill="#EDF0F2",alpha=.5),
  geom_vline(xintercept=.033,colour=grey,linetype="dashed",linewidth=.35))
# Zoom after density estimation so clipped observations still inform the violins.
p2a<-violin_distribution("rf","Residual fraction")+
  scale_y_continuous(breaks=fig2_y_breaks,labels=scales::percent)+
  coord_cartesian(ylim=fig2_y_range,expand=c(bottom=FALSE,left=TRUE,top=FALSE,right=TRUE))
p2b<-violin_distribution("transmission_gain","Residual gain (MSE reduction)")+
  scale_y_continuous(breaks=fig2_y_breaks,labels=scales::label_number(accuracy=.1))+
  coord_cartesian(ylim=fig2_y_range,expand=c(bottom=FALSE,left=TRUE,top=FALSE,right=TRUE))
p2c<-ggplot(m,aes(gaussian_mi_proxy,dataset,colour=dataset))+semantic_reference_region()+
  geom_boxplot(fill="white",width=.5,outlier.shape=NA,linewidth=.45)+geom_point(position=position_jitter(height=.16,seed=42),size=.65,alpha=.6)+
  domain_colours()+guides(colour="none")+
  scale_y_discrete(labels=setNames(domain_names,domains))+
  scale_x_continuous(breaks=fig2_mi_breaks,labels=scales::label_number(accuracy=.01))+
  coord_cartesian(xlim=fig2_mi_range,expand=c(bottom=TRUE,left=FALSE,top=TRUE,right=FALSE))+
  labs(x="Linear Gaussian MI proxy (bits/sample)",y=NULL)
p2d<-ggplot(m,aes(gaussian_mi_proxy,rf,colour=dataset,shape=dataset))+semantic_reference_region()+geom_point(size=.9,alpha=.65)+
  domain_colours()+labs(x="Linear Gaussian MI proxy (bits/sample)",y="Residual fraction",colour=NULL)+
  scale_x_continuous(breaks=fig2_mi_breaks,labels=scales::label_number(accuracy=.01))+
  scale_y_continuous(breaks=fig2_y_breaks,labels=scales::percent)+
  coord_cartesian(xlim=fig2_mi_range,ylim=fig2_y_range,expand=FALSE)+
  scale_shape_manual(values=c(16,17,15),labels=domain_names,name=NULL)+theme(
    legend.position="inside",legend.position.inside=c(.98,.98),
    legend.justification=c(1,1),legend.direction="vertical",
    legend.background=element_blank(),legend.box.background=element_blank())
p2a<-p2a+theme(plot.margin=margin(6,12,4,7))
p2b<-p2b+theme(plot.margin=margin(6,12,4,7))
p2c<-p2c+theme(plot.margin=margin(6,12,4,7))
p2d<-p2d+theme(plot.margin=margin(6,12,4,7))
export_fig(2,list(p2a,p2b,p2c,p2d),c("Residual contribution","Held-out prediction gain","Linear semantic readout","Contribution and readout"),
  list(c(0,2,91,61),c(92,2,91,61),c(0,75,91,61),c(92,75,91,61)),145,"fig2_residual_information")
if(!is.na(selected_figure) && selected_figure==2) {
  write_json(layout_manifest,file.path(out,"layout.json"),auto_unbox=TRUE,pretty=TRUE)
  quit(save="no")
}

if(is.na(selected_figure) || !(selected_figure %in% c(4,5,6))) {
cst$col <- (as.integer(cst$target)-1)*16+cst$rank
defs<-read("construct_definitions")
cst$row<-vapply(seq_len(nrow(cst)),function(i) match(cst$construct[i],defs$construct[defs$dataset==cst$dataset[i]]),integer(1))
cst$heat_sign<-ifelse(cst$shapley_bits>0,"Positive","Non-positive")
construct_names<-c(response_extent="Response extent",paragraph_organization="Paragraph structure",formatted_structure="Formatted structure",explanatory_scaffolding="Explanation",interaction_orientation="Interaction",refusal_compliance="Refusal/compliance",refusal_timing="Refusal timing",safe_redirection="Safe redirection",safety_framing="Safety framing",response_organization="Response organization",reasoning_organization="Reasoning structure",mathematical_formalization="Math formalization",answer_presentation="Answer presentation",self_correction="Self-correction",epistemic_monitoring="Epistemic monitoring",epistemic_stance="Epistemic stance",initial_uncertainty="Initial uncertainty",early_confidence_update="Confidence update",sequence_confidence="Sequence confidence",local_sequence_fit="Local sequence fit")
cst$heat_row<-paste(cst$dataset,cst$construct,sep=":")
cst$heat_row<-factor(cst$heat_row,levels=rev(paste(defs$dataset,defs$construct,sep=":")))
cst$heat_colour<-ifelse(cst$shapley_bits>0,as.character(cst$dataset),"Non-positive")
p3a_full<-ggplot(cst,aes(col,heat_row,fill=heat_colour))+geom_tile(width=1,height=.94)+
  facet_grid(dataset~.,scales="free_y",labeller=labeller(dataset=setNames(domain_names,domains)))+
  geom_vline(xintercept=seq(16.5,112.5,16),colour="white",linewidth=.45)+
  scale_fill_manual(values=c(`Non-positive`="#E8EBEF",pal),breaks=c("Non-positive",domains),labels=c("Non-positive",domain_names),name="Positive Shapley")+
  scale_x_continuous(breaks=seq(8.5,120.5,16),labels=model_names,expand=c(0,0))+
  scale_y_discrete(labels=function(x) construct_names[sub(".*:","",x)],expand=c(0,0))+
  labs(x="Sixteen frozen modules per target (rank order)",y=NULL)+theme(axis.line=element_blank(),axis.ticks=element_blank(),axis.text.x=element_text(size=5.5,angle=25,hjust=1),legend.position="bottom",panel.spacing=unit(2,"mm"),strip.text.y=element_text(angle=0,size=6),strip.placement="outside")
if(is.na(selected_figure)) {
  save_panel(p3a_full,"supp_full_module_construct_map",183,107)
  file.copy(file.path(out,"supp_full_module_construct_map.pdf"),file.path(out,"figS_full_module_construct_map.pdf"),overwrite=TRUE)
}
cst_positive<-transform(cst,positive=as.integer(shapley_bits>0))
positive_share<-aggregate(positive~dataset+target+construct,cst_positive,mean)
positive_count<-aggregate(positive~dataset+target+construct,cst_positive,length)
stopifnot(nrow(positive_share)==240,all(positive_count$positive==16))
write.csv(transform(positive_share,module_count=16),file.path(out,"fig3a_positive_share.csv"),row.names=FALSE)
short_models<-c("L2","L3.2","L3.1","Q3-4","Q2.5","Q3-8","DS-L","DS-Q")
short_constructs<-c(response_extent="Extent",paragraph_organization="Paragraphs",formatted_structure="Formatting",explanatory_scaffolding="Explanation",interaction_orientation="Interaction",refusal_compliance="Refusal",refusal_timing="Refusal timing",safe_redirection="Redirection",safety_framing="Safety framing",response_organization="Organization",reasoning_organization="Reasoning",mathematical_formalization="Math notation",answer_presentation="Answer display",self_correction="Self-correction",epistemic_monitoring="Epistemic",epistemic_stance="Epistemic stance",initial_uncertainty="Initial uncert.",early_confidence_update="Confidence shift",sequence_confidence="Sequence conf.",local_sequence_fit="Local fit")
positive_panel<-function(ds) {
  d<-positive_share[positive_share$dataset==ds,]
  d$construct<-factor(d$construct,levels=rev(defs$construct[defs$dataset==ds]))
  ggplot(d,aes(target,construct,fill=positive))+geom_tile(width=.95,height=.95,colour="white",linewidth=.2)+
    geom_hline(yintercept=5.5,colour="white",linewidth=.65)+coord_fixed(ratio=1)+
    scale_fill_gradientn(limits=c(0,.75),colours=c("#F7FAFC","#D2E8F0","#9FCDE1","#579FC5"),values=scales::rescale(c(0,.25,.5,.75)),breaks=c(0,.25,.5,.75),labels=scales::percent,name="Positive modules / 16",guide=guide_colourbar(barwidth=unit(36,"mm"),barheight=unit(2.5,"mm"),title.position="top",title.hjust=.5))+
    scale_x_discrete(labels=setNames(short_models,models),expand=expansion(add=.5))+
    scale_y_discrete(labels=function(x) short_constructs[x],expand=expansion(add=.5))+
    labs(x=NULL,y=NULL,title=setNames(domain_names,domains)[ds])+
    theme(axis.line=element_blank(),axis.ticks=element_blank(),axis.text.x=element_text(size=5,angle=55,hjust=1),axis.text.y=element_text(size=5),plot.title=element_text(size=7,face="bold",colour=pal[ds],hjust=.5),plot.margin=margin(4,2,2,1))
}
p3a<-wrap_plots(lapply(domains,positive_panel),nrow=1,guides="collect")&theme(legend.position="bottom")
examples<-prep(read.csv(file.path(release,"figure_data/fig3_representative_effects.csv"),check.names=FALSE))
stopifnot(nrow(examples)==12,all(table(examples$dataset,examples$group)==2))
example_effects<-function(g,measure) {
  d<-examples[examples$group==g,]
  d$effect<-d[[paste0(measure,"_bits")]]
  d$low<-d[[paste0(measure,"_low")]]
  d$high<-d[[paste0(measure,"_high")]]
  d$id<-paste(d$dataset,d$target,d$module_key,d$metric,sep=":")
  d$id<-factor(d$id,levels=rev(d$id))
  labels<-setNames(paste0(d$model_label," / M",d$module,"\n",d$metric_label),d$id)
  d$point_colour<-ifelse(d$effect>0,as.character(d$dataset),"Non-positive")
  # Keep paired panels on the same linear scale, including every saved CI endpoint.
  paired<-examples[examples$group==g,]
  extent<-range(0,paired$conditional_low,paired$conditional_high,paired$shapley_low,paired$shapley_high)
  pad<-diff(extent)*.06
  limits<-extent+c(-pad,pad)
  ggplot(d,aes(effect,id,colour=point_colour,shape=dataset))+
    geom_vline(xintercept=0,colour=grey,linetype="dashed",linewidth=.3)+
    geom_errorbar(aes(xmin=low,xmax=high),orientation="y",width=.17,linewidth=.45)+
    geom_point(size=1.65,stroke=.4)+
    facet_grid(dataset~.,scales="free_y",space="free_y",switch="y",
               labeller=labeller(dataset=setNames(domain_names,domains)))+
    scale_colour_manual(values=c(`Non-positive`=grey,pal),guide="none")+
    scale_shape_manual(values=c(16,17,15),guide="none")+
    scale_y_discrete(labels=labels,expand=expansion(add=.55))+
    scale_x_continuous(breaks=pretty(limits,n=4),labels=scales::label_number(accuracy=.01))+
    coord_cartesian(xlim=limits,expand=c(bottom=TRUE,left=FALSE,top=TRUE,right=FALSE))+
    labs(x="Bits/sample",y=NULL)+
    theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),
          axis.text.y=element_text(size=5.5,lineheight=1.15),axis.text.x=element_text(size=5.5),
          axis.title.x=element_text(size=6),strip.text.y.left=element_text(size=5.5,angle=0),
          strip.placement="outside",panel.spacing=unit(2,"mm"),
          plot.margin=margin(4,4,3,2))
}
export_fig(3,list(p3a,example_effects("behavior","conditional"),example_effects("behavior","shapley"),example_effects("monitoring","conditional"),example_effects("monitoring","shapley")),
 c("Positive Shapley share","Behavior: conditional","Behavior: Shapley","Monitoring: conditional","Monitoring: Shapley"),
 list(c(0,1,183,88),c(0,103,90,47),c(93,103,90,47),c(0,161,90,47),c(93,161,90,47)),218,"fig3_conditional_behavior_information")
if(!is.na(selected_figure) && selected_figure==3) {
  write_json(layout_manifest,file.path(out,"layout.json"),auto_unbox=TRUE,pretty=TRUE)
  quit(save="no")
}
}

ret<-do.call(rbind,lapply(domains,function(ds) {d<-m[m$dataset==ds,];data.frame(dataset=ds,ring=1:3,n=c(sum(bool(d$transmission_pass)),sum(bool(d$behavior_pass)),sum(bool(d$three_ring_pass))))}))
ret$x<-ret$ring+(match(ret$dataset,domains)-2)*.07
ret$label_y<-ret$n+ifelse(ret$dataset=="ultrachat",8,ifelse(ret$dataset=="beavertails",ifelse(ret$ring==2,-7,-4),6))
p4a<-ggplot(ret,aes(x,n,group=dataset,colour=dataset,shape=dataset,linetype=dataset))+geom_line(linewidth=.65)+geom_point(size=2)+
  geom_text(aes(y=label_y,label=ifelse(ring==1 & dataset!="mathqa","",n)),size=2.6,show.legend=FALSE)+domain_colours()+
  scale_x_continuous(breaks=1:3,labels=c("Transmission","Association","A/B\nintervention"))+scale_y_continuous(limits=c(0,144))+
  scale_shape_manual(values=c(16,17,15),breaks=domains,labels=domain_names,name=NULL)+scale_linetype_manual(values=c("solid","dashed","dotdash"),breaks=domains,labels=domain_names,name=NULL)+
  labs(x=NULL,y="Modules passing",colour=NULL)+theme(
    legend.position="inside",legend.position.inside=c(.98,.98),
    legend.justification=c(1,1),legend.direction="vertical",
    legend.background=element_blank(),legend.box.background=element_blank(),axis.text.x=element_text(size=5.5))
m$stage<-ifelse(bool(m$three_ring_pass),"Three rings",ifelse(bool(m$behavior_pass),"Rings 1 + 2","Ring 1"))
m$evidence_x<-(as.integer(m$dataset)-1)*17+m$rank
p4b<-ggplot(m,aes(evidence_x,target,fill=stage))+geom_tile(width=.88,height=.82)+
  geom_vline(xintercept=c(17,34),colour=grey,linewidth=.3)+coord_fixed(ratio=1)+
  scale_fill_manual(values=c(`Ring 1`="#E8EBEF",`Rings 1 + 2`="#A8C7BB",`Three rings`="#2D7569"))+
  scale_y_discrete(limits=rev(models),labels=setNames(model_names,models))+
  scale_x_continuous(breaks=(seq_along(domains)-1)*17+8.5,labels=domain_names,expand=expansion(add=.55))+
  labs(x="Dataset",y=NULL,fill=NULL)+
  theme(axis.line=element_blank(),axis.ticks=element_blank(),legend.position="bottom",axis.text.y=element_text(size=6),plot.margin=margin(6,4,4,9))
proj<-nt[nt$metric=="next_token_auto_logit_delta_projection_to_target",]
ab<-merge(proj[proj$screen=="a",],proj[proj$screen=="b",],by=c("dataset","target","module_key"),suffixes=c("_a","_b"))
p4c_full<-ggplot(ab,aes(mean_a,mean_b,colour=dataset))+geom_hline(yintercept=0,colour=grey,linewidth=.3)+geom_vline(xintercept=0,colour=grey,linewidth=.3)+
  geom_abline(slope=1,intercept=0,colour=grey,linetype="dashed",linewidth=.3)+geom_point(aes(shape=bool(three_ring_pass_a)),size=1.5,alpha=.8)+
  domain_colours()+scale_shape_manual(values=c(1,16),labels=c("No","Yes"))+
  labs(x="Screen A: target projection / dose",y="Screen B: target projection / dose",colour=NULL,shape="Three rings")+guides(colour="none")+theme(legend.position="bottom")
zoom_limit<-max(.06,as.numeric(quantile(c(ab$mean_a,ab$mean_b),.90,na.rm=TRUE)))
p4c<-p4c_full+coord_cartesian(xlim=c(-.01,zoom_limit),ylim=c(-.01,zoom_limit))
slopes<-nt[nt$screen=="a" & nt$metric %in% c("next_token_auto_logit_delta_projection_to_target","next_token_auto_logit_margin_delta_toward_target"),]
slopes$metric<-factor(slopes$metric,levels=unique(slopes$metric),labels=c("Target projection","Target margin"))
direction_panel<-function(metric) {
  d<-slopes[slopes$metric==metric,]
  p<-ggplot(d,aes(dataset,mean,fill=dataset,colour=dataset))+geom_hline(yintercept=0,colour=grey,linewidth=.3)+geom_boxplot(width=.5,outlier.shape=NA,linewidth=.3,alpha=.3)+
    geom_point(size=.6,alpha=.55,position=position_jitter(width=.13,seed=42))+scale_fill_manual(values=pal,guide="none")+domain_colours()+guides(colour="none")+scale_x_discrete(labels=c("U","B","M"))+
    labs(x=NULL,y="Change / dose",title=metric)+theme(plot.title=element_text(size=6,hjust=.5),plot.margin=margin(6,2,4,3))
  sigma<-if(metric=="Target projection") .005 else .05
  p+scale_y_continuous(breaks=if(metric=="Target projection") c(-.2,-.02,0,.02,.2,.6) else c(-5,-.5,0,.5,5,20))+
    coord_trans(y=scales::pseudo_log_trans(sigma=sigma,base=10))+
    labs(y="Change / dose (asinh)")
}
p4d<-direction_panel("Target projection")+direction_panel("Target margin")+plot_layout(ncol=2)
export_fig(4,list(p4a,p4b,p4c,p4d),c("Three-ring progression","Same-module evidence","Independent next-token screens","Directional dose effects"),
 list(c(0,2,53,73),c(0,89,183,52),c(55,2,59,73),c(116,2,67,73)),150,"fig4_intervention_dose_response")
if(!is.na(selected_figure) && selected_figure==4) {
  write_json(layout_manifest,file.path(out,"layout.json"),auto_unbox=TRUE,pretty=TRUE)
  quit(save="no")
}

# All 24 slots use the completed fresh evaluation, with no mixed-run estimates.
report_slots<-prep(expand.grid(dataset=domains,target=models,stringsAsFactors=FALSE))
report_slots<-merge(report_slots,rc,by=c("dataset","target"),all.x=TRUE,sort=FALSE)
report_slots<-report_slots[order(report_slots$dataset,report_slots$target),]
stopifnot(nrow(report_slots)==24,!anyDuplicated(report_slots[c("dataset","target")]))
report_slots$row<-(3-as.integer(report_slots$dataset))*9+9-as.integer(report_slots$target)
report_slots$label<-setNames(model_names,models)[as.character(report_slots$target)]
first_model<-report_slots$target==models[1]
report_slots$label[first_model]<-paste(setNames(domain_names,domains)[as.character(report_slots$dataset[first_model])],report_slots$label[first_model],sep=" / ")
stopifnot(all(!is.na(report_slots$status)),all(report_slots$status=="complete"),nrow(rn)==384)
report_slots$source_status<-"Observed"
report_observed<-report_slots
row_labels<-setNames(report_slots$label,report_slots$row)
report_y<-function(labels=TRUE) scale_y_continuous(breaks=report_slots$row,
  labels=if(labels) row_labels[as.character(report_slots$row)] else NULL,
  limits=c(.35,26.65),expand=c(0,0))
report_theme<-function(labels=TRUE) theme(legend.position="bottom",legend.direction="horizontal",
  axis.text.y=if(labels) element_text(size=5.5) else element_blank(),
  axis.ticks.y=element_blank(),axis.line.y=element_blank(),
  plot.margin=margin(6,5,4,7),legend.key.size=unit(2.5,"mm"))
report_background<-function(xmin,xmax) list(
  geom_hline(yintercept=c(9,18),colour="#D7DCE1",linewidth=.25))
p5a<-ggplot(report_observed,aes(y=row))+report_background(.4,1)+
 geom_vline(xintercept=.5,linetype="dashed",colour=grey,linewidth=.3)+
 geom_segment(aes(x=primary_auc,xend=primary_semantic_label_auc,yend=row),colour=grey,linewidth=.4)+
 geom_point(aes(x=primary_semantic_label_auc,colour="Semantic predictor"),size=1.25)+
 geom_point(aes(x=primary_auc,colour="Direct report"),size=1.25)+
 scale_colour_manual(values=c(`Semantic predictor`=warm,`Direct report`=blue))+
 scale_x_continuous(limits=c(.4,1),breaks=seq(.4,1,.1),expand=c(0,0))+report_y()+
 labs(x="Held-out activation-label AUC",y=NULL,colour=NULL)+report_theme()
p5b<-ggplot(report_observed,aes(rho_screen,row,colour=dataset))+report_background(-.22,.22)+
 geom_vline(xintercept=0,colour=grey,linewidth=.3)+
 geom_segment(aes(x=0,xend=rho_screen,yend=row),linewidth=.4,alpha=.6)+
 geom_point(size=1.3)+domain_colours()+report_y(FALSE)+
 scale_x_continuous(limits=c(-.22,.22),breaks=seq(-.2,.2,.1),expand=c(0,0))+
 labs(x="Semantic predictability vs report AUC (rho)",y=NULL,colour=NULL)+report_theme(FALSE)

# The prediction target here is report log-odds, not the activation label.
rn_report<-merge(rn,report_observed[,c("dataset","target","row")],by=c("dataset","target"),sort=FALSE)
gain_details<-lapply(rn_report$conditional_report_gain,fromJSON)
rn_report$semantic_report_mse<-vapply(gain_details,function(x) x$semantic_mse,numeric(1))
rn_report$joint_report_mse<-vapply(gain_details,function(x) x$semantic_plus_activation_mse,numeric(1))
stopifnot(all(is.finite(rn_report$semantic_report_mse)),all(rn_report$semantic_report_mse>0),
  all(is.finite(rn_report$joint_report_mse)),all(rn_report$joint_report_mse>=0))
rn_report$mse_ratio<-rn_report$joint_report_mse/rn_report$semantic_report_mse
rn_report$row_group<-rn_report$row+ifelse(rn_report$group=="winners",.18,-.18)
report_centers<-aggregate(mse_ratio~dataset+target+group+row_group,rn_report,median)
ratio_limits<-range(c(.88,1.08,rn_report$mse_ratio)) + c(-.01,.01)
p5c<-ggplot(rn_report,aes(mse_ratio,row_group,colour=dataset,shape=group))+
 report_background(ratio_limits[1],ratio_limits[2])+geom_vline(xintercept=1,colour=grey,linetype="dashed",linewidth=.3)+
 geom_point(position=position_jitter(height=.07,width=0,seed=42),size=.7,alpha=.65)+
 geom_point(data=report_centers,size=1.6,stroke=.45)+domain_colours()+
 scale_shape_manual(values=c(winners=16,matched_controls=1),breaks=c("winners","matched_controls"),
   labels=c("Selected neurons","Matched controls"),name=NULL)+guides(colour="none")+report_y()+
 scale_x_continuous(limits=ratio_limits,breaks=seq(.8,1.1,.1),expand=c(0,0))+
 labs(x="Report MSE ratio: (semantic + activation) / semantic",y=NULL)+report_theme()
flip_rows<-rn_report[rn_report$group=="winners" & is.finite(rn_report$target_minus_control_movement),]
flip_summary<-do.call(rbind,lapply(split(flip_rows,interaction(flip_rows$dataset,flip_rows$target,drop=TRUE)),function(d) {
  q<-quantile(d$target_minus_control_movement,c(.25,.5,.75),names=FALSE)
  data.frame(dataset=d$dataset[1],target=d$target[1],row=d$row[1],low=q[1],median=q[2],high=q[3],n=nrow(d))
}))
flip_limits<-range(c(-.015,.025,flip_rows$target_minus_control_movement))+c(-.002,.002)
p5d<-ggplot(flip_rows,aes(target_minus_control_movement,row,colour=dataset))+
 report_background(flip_limits[1],flip_limits[2])+geom_vline(xintercept=0,colour=grey,linewidth=.3)+
 geom_point(aes(shape="Neuron"),size=.7,alpha=.65,position=position_jitter(height=.13,width=0,seed=42))+
 geom_segment(data=flip_summary,aes(x=low,xend=high,y=row,yend=row),inherit.aes=FALSE,linewidth=.55,colour="#343A43")+
 geom_point(data=flip_summary,aes(x=median,y=row,shape="Median"),inherit.aes=FALSE,size=1.6,colour="#343A43")+
 domain_colours()+report_y(FALSE)+guides(colour="none")+
 scale_shape_manual(values=c(Neuron=16,Median=18),breaks=c("Neuron","Median"),name=NULL)+
 scale_x_continuous(limits=flip_limits,breaks=seq(-.04,.08,.02),expand=c(0,0))+
 labs(x="Target minus control report movement (log-odds)",y=NULL)+report_theme(FALSE)
write.csv(report_slots,file.path(out,"fig5_condition_slots.csv"),row.names=FALSE)
write.csv(rn_report,file.path(out,"fig5_neuron_predictions.csv"),row.names=FALSE)
write.csv(flip_summary,file.path(out,"fig5_flip_summary.csv"),row.names=FALSE)
write_json(list(planned_slots=24,observed_conditions=nrow(report_observed),pending_slots=0,
  input_conditions="03_source_data/strict_current/report_conditions.csv",input_neurons="03_source_data/strict_current/report_neurons.csv",
  current_run="neuron_report_fresh24_20261008_104456",new_run_results_loaded=TRUE,simulated_values=FALSE,
  panel_c_target="report_high_logodds",panel_c_formula="semantic_plus_activation_mse / semantic_mse",
  panel_c_estimation="five_fold_out_of_fold",panel_d_units="report log-odds",
  panel_d_summary="median and interquartile range across selected neurons"),
  file.path(out,"fig5_data_contract.json"),auto_unbox=TRUE,pretty=TRUE)
export_fig(5,list(p5a,p5b,p5c,p5d),c("Direct reports and semantic prediction","Semantic-report association","Activation contribution to reports","Activation-flip response"),
 list(c(0,2,110,90),c(114,2,69,90),c(0,108,110,90),c(114,108,69,90)),208,"fig5_neuron_direct_report")
if(!is.na(selected_figure) && selected_figure==5) {
  write_json(layout_manifest,file.path(out,"layout.json"),auto_unbox=TRUE,pretty=TRUE)
  quit(save="no")
}

tr$label<-paste(setNames(domain_names,domains)[as.character(tr$dataset)],setNames(model_names,models)[as.character(tr$target)],sub("rank_","R",sub("_k32.*","",tr$module_key)),sep=" / ")
tr$label<-factor(tr$label,levels=rev(tr$label))
tr$kind<-ifelse(tr$construct=="initial_uncertainty","Prompt-end concentration","Response / trajectory")
tr$significant<-bool(tr$direction_consistent) & tr$directional_one_sided_p<.05
tc<-read("trajectory_components")
unstable<-unique(tc[tc$baseline_sd<1e-4,c("dataset","target","module_key")])
unstable_keys<-with(unstable,paste(dataset,target,module_key))
tr_plot<-tr[!with(tr,paste(dataset,target,module_key)) %in% unstable_keys,]
p6a<-ggplot(tr_plot,aes(dataset,expectation_aligned_effect,fill=dataset,colour=dataset))+
 geom_hline(yintercept=0,colour=grey,linewidth=.3)+
 geom_boxplot(width=.5,alpha=.3,outlier.shape=NA,linewidth=.35)+
 geom_point(aes(shape=significant),position=position_jitter(width=.17,seed=42),size=1,alpha=.8)+
 scale_fill_manual(values=pal,guide="none")+domain_colours()+guides(colour="none")+
 scale_shape_manual(values=c(`FALSE`=1,`TRUE`=16),labels=c("Other","Directional p < 0.05"),name=NULL)+
 scale_x_discrete(labels=c("U","B","M"))+labs(x=NULL,y="Association-aligned change / dose")+theme(legend.position="bottom",legend.text=element_text(size=5.5),legend.key.size=unit(2,"mm"))
tc<-tc[!with(tc,paste(dataset,target,module_key)) %in% unstable_keys,]
tc$aligned<-tc$mean_standardized_positive_minus_negative
curve<-aggregate(aligned~dataset+target+module_key+construct+dose,tc,mean)
curve<-merge(curve,tr[,c("dataset","target","module_key","expected_direction")],by=c("dataset","target","module_key"))
curve$aligned<-ifelse(curve$expected_direction=="positive",1,-1)*curve$aligned
defs<-read("construct_definitions")
curve$group<-ifelse(curve$construct %in% defs$construct[defs$group=="monitoring"],"Monitoring","Behavior")
# Each module has one vote per dose, independent of its effect magnitude.
curve$direction_aligned<-as.integer(curve$aligned>0)
aligned_counts<-aggregate(direction_aligned~dataset+group+dose,curve,sum)
module_counts<-aggregate(direction_aligned~dataset+group+dose,curve,length)
names(module_counts)[4]<-"evaluable_modules"
dose_summary<-merge(aligned_counts,module_counts,by=c("dataset","group","dose"))
dose_summary$aligned_fraction<-dose_summary$direction_aligned/dose_summary$evaluable_modules
stopifnot(nrow(dose_summary)==18,all(is.finite(curve$aligned)),
          !anyDuplicated(curve[c("dataset","target","module_key","dose")]))
write.csv(dose_summary,file.path(out,"fig6b_dose_direction_consistency.csv"),row.names=FALSE)
dose_summary$dataset<-factor(dose_summary$dataset,levels=rev(domains))
dose_summary$group<-factor(dose_summary$group,levels=c("Behavior","Monitoring"))
dose_summary$dose_label<-factor(dose_summary$dose,levels=c(.25,.5,1),labels=c("0.25","0.5","1.0"))
dose_summary$count_label<-paste0(dose_summary$direction_aligned,"/",dose_summary$evaluable_modules)
p6b<-ggplot(dose_summary,aes(dose_label,dataset,fill=aligned_fraction))+
 geom_tile(width=.94,height=.86,colour="white",linewidth=.4)+
 geom_text(aes(label=count_label,colour=aligned_fraction>.7),size=7/ggplot2::.pt)+
 scale_colour_manual(values=c(`FALSE`="#343A43",`TRUE`="white"),guide="none")+
 scale_fill_gradientn(colours=c("#EDF3F6","#AACFDF",blue),limits=c(0,1),breaks=c(0,.5,1),
   labels=scales::percent,name="Aligned modules",
   guide=guide_colourbar(barwidth=unit(32,"mm"),barheight=unit(2.5,"mm"),title.position="top",title.hjust=.5))+
 facet_wrap(~group,nrow=1)+scale_y_discrete(labels=setNames(domain_names,domains))+
 labs(x="Dose (module SD)",y=NULL)+theme(legend.position="bottom",axis.line=element_blank(),
   axis.ticks=element_blank(),panel.spacing=unit(4,"mm"),strip.text=element_text(size=7,face="bold"))
examples<-fromJSON(file.path(data_dir,"response_examples.json"),simplifyVector=FALSE)
word_layout<-list(); titles_layout<-list()
cairo_pdf(file.path(out,"text_measurement.pdf"),width=183/25.4,height=74/25.4,family="Arial")
for(i in seq_along(examples)) {
  e<-examples[[i]];column<-((i-1)%%3); row<-floor((i-1)/3)
  left<-3+column*61; top<-10+row*37; x<-left; y<-top
  titles_layout[[i]]<-data.frame(x=left,y=top-4,text=c("Reverse (-)","Baseline","Forward (+)")[column+1])
  words<-unlist(e$words)
  for(j in seq_along(words)) {
    w<-words[j]
    g<-grid::textGrob(paste0(w," "),gp=grid::gpar(fontfamily="Arial",fontsize=7.2))
    width<-grid::convertWidth(grid::grobWidth(g),"mm",valueOnly=TRUE)
    if(x+width>left+55) {x<-left;y<-y+3.6}
    word_layout[[length(word_layout)+1]]<-data.frame(x=x,y=y,width=width,text=w,colour=ifelse((j-1)%in%unlist(e$changed_word_indices),blue,"#343A43"),case=row+1)
    x<-x+width
  }
}
dev.off()
words<-do.call(rbind,word_layout);titles<-do.call(rbind,titles_layout)
case_headers<-data.frame(x=6,y=c(1,38),text=c("BeaverTails / Llama2-7B / rank 11: refusal wording","MathQA / DS-Llama-8B / rank 12: mathematical exposition"))
p6c<-ggplot()+geom_text(data=words,aes(x=x,y=74-y,label=text,colour=colour),hjust=0,vjust=1,size=7.2/ggplot2::.pt)+
 geom_text(data=titles,aes(x=x,y=74-y,label=text),hjust=0,vjust=1,size=7/ggplot2::.pt,fontface="bold",colour=blue)+
 geom_text(data=case_headers,aes(x=x,y=74-y,label=text),hjust=0,vjust=1,size=7/ggplot2::.pt,fontface="bold")+
 scale_colour_identity()+coord_cartesian(xlim=c(0,183),ylim=c(0,74),expand=FALSE,clip="off")+theme_void()+theme(plot.margin=margin(0,0,0,0))
write_json(list(words=words,titles=titles,headers=case_headers,font_size=7.2,height=74),file.path(out,"case_layout.json"),auto_unbox=TRUE,pretty=TRUE)
export_fig(6,list(p6a,p6b,p6c),c("Primary construct dose responses","Dose-wise direction consistency","Real branch excerpts (changed words in blue)"),
 list(c(0,2,70,69),c(75,2,108,69),c(0,83,183,74)),166,"fig6_persistent_generation")
if(!is.na(selected_figure)) {
  write_json(layout_manifest,file.path(out,"layout.json"),auto_unbox=TRUE,pretty=TRUE)
  quit(save="no")
}

replay<-fromJSON(file.path(data_dir,"module_self_report_summary.json"))$conditions
replay<-replay[replay$status=="complete",]
replay$label<-paste(replay$dataset,replay$target)
diag1<-ggplot(replay,aes(semantic_only_label_auc,natural_report_auc))+
 geom_abline(slope=1,intercept=0,linetype="dashed",colour=grey)+
 geom_point(colour=blue,size=1.7)+labs(x="Semantic label AUC",y="Module report AUC")
diag2<-ggplot(replay,aes(baseline_answer_option_mass_mean,reorder(label,baseline_answer_option_mass_mean)))+
 geom_point(colour=warm,size=1.5)+scale_x_log10()+labs(x="Total A/B option probability (log scale)",y=NULL)+
 theme(axis.text.y=element_text(size=5.5))
si_dir<-file.path(out,"supplementary")
dir.create(si_dir,recursive=TRUE,showWarnings=FALSE)
ggsave(file.path(si_dir,"module_report_diagnostics.pdf"),diag1+diag2+plot_layout(widths=c(1,1.5)),width=170,height=100,units="mm",device=cairo_pdf)

write_json(layout_manifest,file.path(out,"layout.json"),auto_unbox=TRUE,pretty=TRUE)
