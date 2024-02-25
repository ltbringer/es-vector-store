{{- define "elasticsearch.fullname" -}}
{{- printf "%s-%s" .Release.Name "elasticsearch" -}}
{{- end -}}

{{- define "elasticsearch.name" -}}
{{- printf "%s" "elasticsearch" -}}
{{- end -}}

{{- define "elasticsearch.chart" -}}
{{- printf "%s" "elasticsearch-chart" -}}
{{- end -}}
